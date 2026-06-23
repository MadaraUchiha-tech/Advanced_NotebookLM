import { RankedDocument } from "./reranker";

/**
 * Token Manager — Stage 5 of the Advanced RAG Pipeline
 *
 * LLMs have fixed context windows. Stuffing too many chunks causes:
 * 1. API errors (exceeding max tokens)
 * 2. "Lost in the middle" — models ignore context in the middle of long prompts
 * 3. Higher cost and latency
 *
 * This module:
 * - Estimates token counts using a word-based heuristic (avoids WASM issues with tiktoken)
 * - Allocates a budget: system prompt + context + user query + response reserve
 * - Greedily selects the highest-ranked chunks that fit within the budget
 * - Returns the optimized context string
 */

// GPT-4o-mini context limits — we use a conservative budget
const MODEL_CONTEXT_LIMIT = 128_000;
const TARGET_CONTEXT_TOKENS = 6_000; // Keep context concise for quality
const SYSTEM_PROMPT_RESERVE = 500; // Tokens reserved for system instructions
const RESPONSE_RESERVE = 1_500; // Tokens reserved for the model's response
const QUERY_RESERVE = 200; // Tokens reserved for user query

const AVAILABLE_CONTEXT_BUDGET =
  TARGET_CONTEXT_TOKENS - SYSTEM_PROMPT_RESERVE - RESPONSE_RESERVE - QUERY_RESERVE;

export interface TokenBudgetResult {
  selectedDocuments: RankedDocument[];
  droppedCount: number;
  totalTokensUsed: number;
  budgetLimit: number;
  contextString: string;
}

/**
 * Estimate token count using word-based heuristic.
 * Average English word ≈ 1.3 tokens. This avoids WASM/native dependency issues
 * with tiktoken while remaining reasonably accurate.
 */
export function estimateTokens(text: string): number {
  // Split on whitespace and punctuation, count resulting tokens
  // Average ratio: ~1.3 tokens per word for English text
  const words = text.split(/\s+/).filter((w) => w.length > 0);
  return Math.ceil(words.length * 1.3);
}

/**
 * Select the best-fitting documents within the token budget.
 * Documents are already sorted by relevance (highest first).
 * We greedily add documents until the budget is exhausted.
 */
export function fitToTokenBudget(
  rankedDocs: RankedDocument[],
  budgetOverride?: number
): TokenBudgetResult {
  const budget = budgetOverride || AVAILABLE_CONTEXT_BUDGET;
  const selected: RankedDocument[] = [];
  let tokensUsed = 0;

  for (const rankedDoc of rankedDocs) {
    const docTokens = estimateTokens(rankedDoc.document.pageContent);

    if (tokensUsed + docTokens <= budget) {
      selected.push(rankedDoc);
      tokensUsed += docTokens;
    } else {
      // Try to fit a truncated version if the doc is partially useful
      const remainingBudget = budget - tokensUsed;
      if (remainingBudget > 100) {
        // At least 100 tokens worth including
        const truncatedContent = truncateToTokens(
          rankedDoc.document.pageContent,
          remainingBudget
        );
        selected.push({
          ...rankedDoc,
          document: {
            ...rankedDoc.document,
            pageContent: truncatedContent + "\n[...truncated]",
          },
        });
        tokensUsed += estimateTokens(truncatedContent);
      }
      break; // Budget exhausted
    }
  }

  // Build the formatted context string
  const contextString = selected
    .map((rd, i) => {
      const source = rd.document.metadata?.source || "document";
      const page =
        rd.document.metadata?.page ||
        rd.document.metadata?.loc?.pageNumber ||
        "N/A";
      const score = rd.relevanceScore.toFixed(1);
      return `[${i + 1}] Source: ${source}, Page: ${page}, Relevance: ${score}/10\n${rd.document.pageContent}`;
    })
    .join("\n\n---\n\n");

  return {
    selectedDocuments: selected,
    droppedCount: rankedDocs.length - selected.length,
    totalTokensUsed: tokensUsed,
    budgetLimit: budget,
    contextString,
  };
}

/**
 * Truncate text to approximately the target token count.
 */
function truncateToTokens(text: string, targetTokens: number): string {
  const words = text.split(/\s+/);
  // ~1.3 tokens per word, so targetTokens / 1.3 ≈ words to keep
  const wordsToKeep = Math.floor(targetTokens / 1.3);
  return words.slice(0, wordsToKeep).join(" ");
}
