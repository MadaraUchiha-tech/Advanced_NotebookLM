import OpenAI from "openai";
import { Document } from "@langchain/core/documents";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const OPENAI_BASE_URL =
  process.env.OPENAI_BASE_URL || "https://models.github.ai/inference";

const openai = new OpenAI({ apiKey: OPENAI_API_KEY, baseURL: OPENAI_BASE_URL });

export interface RankedDocument {
  document: Document;
  relevanceScore: number;
  reasoning: string;
}

export interface RerankResult {
  rankedDocuments: RankedDocument[];
  removedCount: number;
  totalScored: number;
}

const RELEVANCE_THRESHOLD = 4; // Out of 10 — chunks below this are filtered out

/**
 * LLM-based Cross-Encoder Reranker — Stage 4
 *
 * Vector similarity search (bi-encoder) is fast but coarse. A cross-encoder
 * looks at the (query, document) pair together and produces a much more
 * accurate relevance judgment.
 *
 * We use GPT-4o-mini as a lightweight cross-encoder surrogate:
 * - Score each retrieved chunk 0-10 for relevance to the query
 * - Filter out low-scoring chunks (< threshold)
 * - Re-sort by relevance score (highest first)
 *
 * This dramatically improves precision: even if vector search returns some
 * irrelevant chunks, the reranker pushes them out.
 */
export async function rerankDocuments(
  query: string,
  documents: Document[]
): Promise<RerankResult> {
  if (documents.length === 0) {
    return { rankedDocuments: [], removedCount: 0, totalScored: 0 };
  }

  // Score documents in parallel batches of 5 for efficiency
  const batchSize = 5;
  const scoredDocs: RankedDocument[] = [];

  for (let i = 0; i < documents.length; i += batchSize) {
    const batch = documents.slice(i, i + batchSize);
    const batchResults = await Promise.all(
      batch.map((doc) => scoreDocument(query, doc))
    );
    scoredDocs.push(...batchResults);
  }

  // Sort by relevance score (highest first)
  scoredDocs.sort((a, b) => b.relevanceScore - a.relevanceScore);

  // Filter out low-relevance chunks
  const filtered = scoredDocs.filter(
    (d) => d.relevanceScore >= RELEVANCE_THRESHOLD
  );

  return {
    rankedDocuments: filtered,
    removedCount: scoredDocs.length - filtered.length,
    totalScored: scoredDocs.length,
  };
}

/**
 * Score a single (query, document) pair for relevance.
 */
async function scoreDocument(
  query: string,
  doc: Document
): Promise<RankedDocument> {
  const systemPrompt = `You are a relevance judge. Score how relevant a document passage is to a user's query.

Score on a scale of 0-10:
- 0-3: Not relevant — passage discusses unrelated topics
- 4-5: Marginally relevant — passage touches on the topic but doesn't address the query
- 6-7: Relevant — passage contains information useful for answering the query
- 8-9: Highly relevant — passage directly answers or strongly supports answering the query
- 10: Perfect match — passage is exactly what's needed to answer the query

Respond in valid JSON:
{
  "score": <number 0-10>,
  "reasoning": "One sentence explaining the score"
}`;

  const userPrompt = `Query: "${query}"

Document passage:
"""
${doc.pageContent.substring(0, 1500)}
"""`;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: 0.0,
      response_format: { type: "json_object" },
      max_tokens: 100,
    });

    const content = response.choices[0].message.content || "{}";
    const parsed = JSON.parse(content);

    return {
      document: doc,
      relevanceScore: Math.min(10, Math.max(0, parsed.score || 0)),
      reasoning: parsed.reasoning || "No reasoning",
    };
  } catch (error) {
    console.error("Reranking score failed for document:", error);
    // On failure, give a neutral score so the document isn't dropped
    return {
      document: doc,
      relevanceScore: 5,
      reasoning: "Scoring failed — assigned neutral score",
    };
  }
}
