import OpenAI from "openai";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const OPENAI_BASE_URL =
  process.env.OPENAI_BASE_URL || "https://models.github.ai/inference";

const openai = new OpenAI({ apiKey: OPENAI_API_KEY, baseURL: OPENAI_BASE_URL });

export interface QueryRewriteResult {
  originalQuery: string;
  rewrittenQueries: string[];
  reasoning: string;
}

/**
 * Query Rewriter — Stage 1 of the Advanced RAG Pipeline
 *
 * Takes a raw user query and produces 2-3 semantically diverse reformulations
 * optimized for vector similarity search. This compensates for poor user
 * phrasing, ambiguity, or overly specific/vague language.
 *
 * Techniques:
 * - Expansion: adds synonyms and related terms
 * - Specification: makes vague queries more precise
 * - Decomposition hint: splits compound questions
 */
export async function rewriteQuery(
  query: string
): Promise<QueryRewriteResult> {
  const systemPrompt = `You are a search query optimizer for a document retrieval system.

Given a user's natural language question, generate 2-3 alternative search queries that:
1. Use different vocabulary/synonyms to capture the same intent
2. Are more specific and precise than the original
3. Would match relevant passages in a technical document

Rules:
- Keep each query concise (under 30 words)
- Preserve the original intent — do NOT change the meaning
- Include the key entities and concepts from the original query
- If the query is already well-formed, still provide at least one variation

Respond in valid JSON format:
{
  "rewrittenQueries": ["query1", "query2", "query3"],
  "reasoning": "Brief explanation of why these reformulations help"
}`;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: query },
      ],
      temperature: 0.3,
      response_format: { type: "json_object" },
    });

    const content = response.choices[0].message.content || "{}";
    const parsed = JSON.parse(content);

    return {
      originalQuery: query,
      rewrittenQueries: parsed.rewrittenQueries || [query],
      reasoning: parsed.reasoning || "No reasoning provided",
    };
  } catch (error) {
    console.error("Query rewrite failed, using original query:", error);
    return {
      originalQuery: query,
      rewrittenQueries: [query],
      reasoning: "Rewrite failed — falling back to original query",
    };
  }
}
