import OpenAI from "openai";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const OPENAI_BASE_URL =
  process.env.OPENAI_BASE_URL || "https://models.github.ai/inference";

const openai = new OpenAI({ apiKey: OPENAI_API_KEY, baseURL: OPENAI_BASE_URL });

export interface SubQueryResult {
  originalQuery: string;
  needsDecomposition: boolean;
  subQueries: string[];
  reasoning: string;
}

/**
 * Sub-Query Decomposer — Stage 3 of the Advanced RAG Pipeline
 *
 * Complex questions often require information from multiple parts of a document.
 * For example: "Compare the revenue in Q1 vs Q3 and explain the difference"
 * requires retrieving Q1 data, Q3 data, AND analysis sections.
 *
 * This module:
 * 1. Detects whether a query is "simple" (single-fact) or "complex" (multi-hop)
 * 2. If complex, decomposes it into 2-4 atomic sub-queries
 * 3. Each sub-query is independently retrievable
 *
 * The pipeline runs retrieval for each sub-query, then merges and deduplicates
 * the results before generation.
 */
export async function decomposeQuery(
  query: string
): Promise<SubQueryResult> {
  const systemPrompt = `You are a query analysis expert. Determine if a user's question is simple or complex, and decompose complex questions into atomic sub-queries.

A question is COMPLEX if it:
- Asks about multiple distinct facts or entities
- Requires comparison between items
- Has multiple parts joined by "and", "or", "vs", "compared to"
- Needs information from different sections of a document
- Involves cause-and-effect reasoning across topics

A question is SIMPLE if it:
- Asks about a single fact, definition, or concept
- Can be answered from a single passage
- Is a straightforward lookup

Respond in valid JSON:
{
  "needsDecomposition": true/false,
  "subQueries": ["sub-query-1", "sub-query-2"],
  "reasoning": "Why this query was/wasn't decomposed"
}

Rules:
- Maximum 4 sub-queries
- Each sub-query should be self-contained and independently searchable
- Sub-queries should collectively cover the full scope of the original question
- For simple questions, return the original query as the only sub-query`;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: query },
      ],
      temperature: 0.1,
      response_format: { type: "json_object" },
    });

    const content = response.choices[0].message.content || "{}";
    const parsed = JSON.parse(content);

    return {
      originalQuery: query,
      needsDecomposition: parsed.needsDecomposition || false,
      subQueries: parsed.subQueries || [query],
      reasoning: parsed.reasoning || "No reasoning provided",
    };
  } catch (error) {
    console.error("Sub-query decomposition failed:", error);
    return {
      originalQuery: query,
      needsDecomposition: false,
      subQueries: [query],
      reasoning: "Decomposition failed — using original query",
    };
  }
}
