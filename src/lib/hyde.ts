import OpenAI from "openai";
import { OpenAIEmbeddings } from "@langchain/openai";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const OPENAI_BASE_URL =
  process.env.OPENAI_BASE_URL || "https://models.github.ai/inference";

const openai = new OpenAI({ apiKey: OPENAI_API_KEY, baseURL: OPENAI_BASE_URL });

const embeddings = new OpenAIEmbeddings({
  modelName: "text-embedding-3-large",
  apiKey: OPENAI_API_KEY,
  configuration: { baseURL: OPENAI_BASE_URL },
});

export interface HyDEResult {
  originalQuery: string;
  hypotheticalDocument: string;
  embedding: number[];
}

/**
 * HyDE — Hypothetical Document Embeddings (Stage 2)
 *
 * Instead of embedding the user's *question*, we ask the LLM to generate a
 * hypothetical answer passage — the kind of text that *would* appear in a
 * relevant document. We then embed *that* passage for similarity search.
 *
 * Why this works:
 * - Questions and answers live in different embedding regions
 * - A hypothetical answer is lexically closer to real document chunks
 * - This bridges the "query–document vocabulary mismatch" problem
 *
 * Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without
 * Relevance Labels" (2022)
 */
export async function generateHyDE(query: string): Promise<HyDEResult> {
  const systemPrompt = `You are a technical document author. Given a question, write a short passage (100-200 words) that would appear in a document that answers this question.

Rules:
- Write in a factual, encyclopedic tone as if this passage comes from a real document
- Include specific details, terminology, and concepts relevant to the question
- Do NOT frame it as an answer — frame it as content from a source document
- Do NOT start with "This document..." or "According to..." — just write the content directly
- Be informative and dense with relevant keywords`;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: query },
      ],
      temperature: 0.5,
      max_tokens: 300,
    });

    const hypotheticalDocument =
      response.choices[0].message.content || query;

    // Embed the hypothetical document instead of the raw query
    const embedding = await embeddings.embedQuery(hypotheticalDocument);

    return {
      originalQuery: query,
      hypotheticalDocument,
      embedding,
    };
  } catch (error) {
    console.error("HyDE generation failed, falling back to query embedding:", error);

    // Fallback: embed the original query
    const embedding = await embeddings.embedQuery(query);

    return {
      originalQuery: query,
      hypotheticalDocument: query,
      embedding,
    };
  }
}
