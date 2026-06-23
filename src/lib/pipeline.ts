import { Document } from "@langchain/core/documents";
import { rewriteQuery, QueryRewriteResult } from "./queryRewriter";
import { generateHyDE, HyDEResult } from "./hyde";
import { decomposeQuery, SubQueryResult } from "./subQueryDecomposer";
import { multiQueryRetrieve, hydeRetrieve, openai } from "./rag";
import { rerankDocuments, RerankResult, RankedDocument } from "./reranker";
import { fitToTokenBudget, TokenBudgetResult } from "./tokenManager";
import { judgeAnswer, JudgeResult } from "./judge";

// ── Types ────────────────────────────────────────────────────────────

export interface PipelineStageStatus {
  name: string;
  status: "pending" | "running" | "completed" | "skipped" | "failed";
  durationMs?: number;
  detail?: string;
}

export interface PipelineTrace {
  stages: PipelineStageStatus[];
  queryRewrite?: QueryRewriteResult;
  hyde?: HyDEResult;
  subQueryDecomposition?: SubQueryResult;
  retrievedCount?: number;
  rerank?: {
    totalScored: number;
    removedCount: number;
    topScores: number[];
  };
  tokenBudget?: {
    tokensUsed: number;
    budgetLimit: number;
    chunksUsed: number;
    chunksDropped: number;
  };
  judge?: JudgeResult;
  retryCount: number;
}

export interface PipelineResult {
  answer: string;
  sources: { source: string; page: string | number; relevanceScore: number }[];
  trace: PipelineTrace;
  totalDurationMs: number;
}

const MAX_RETRIES = 1; // Corrective RAG: retry once on judge failure

// ── Pipeline Orchestrator ────────────────────────────────────────────

/**
 * Advanced RAG Pipeline — Orchestrator
 *
 * Wires together all 7 stages of the advanced RAG pipeline:
 *
 * 1. Query Rewriting → optimize the query for retrieval
 * 2. HyDE → generate hypothetical doc embedding
 * 3. Sub-Query Decomposition → split complex queries
 * 4. Multi-source Retrieval → search with all query variants + HyDE
 * 5. Cross-Encoder Reranking → score and filter chunks
 * 6. Token Management → fit within context window
 * 7. Generation + Judge → generate answer, evaluate quality
 *
 * Corrective RAG loop: if the judge fails the answer, retry with
 * the judge's suggestions incorporated into a rewritten query.
 */
export async function runAdvancedPipeline(
  query: string,
  collectionName: string
): Promise<PipelineResult> {
  const pipelineStart = Date.now();

  const trace: PipelineTrace = {
    stages: [
      { name: "Query Rewriting", status: "pending" },
      { name: "HyDE Generation", status: "pending" },
      { name: "Sub-Query Decomposition", status: "pending" },
      { name: "Multi-Source Retrieval", status: "pending" },
      { name: "Cross-Encoder Reranking", status: "pending" },
      { name: "Token Management", status: "pending" },
      { name: "Generation", status: "pending" },
      { name: "LLM Judge", status: "pending" },
    ],
    retryCount: 0,
  };

  let currentQuery = query;
  let finalAnswer = "";
  let finalSources: PipelineResult["sources"] = [];
  let contextForJudge = "";

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    if (attempt > 0) {
      trace.retryCount = attempt;
      // Reset stages for retry
      trace.stages.forEach((s) => {
        s.status = "pending";
        s.durationMs = undefined;
        s.detail = undefined;
      });
    }

    // ── Stage 1: Query Rewriting ──────────────────────────────────
    const stageRewrite = trace.stages[0];
    stageRewrite.status = "running";
    const rewriteStart = Date.now();

    let rewriteResult: QueryRewriteResult;
    try {
      rewriteResult = await rewriteQuery(currentQuery);
      stageRewrite.status = "completed";
      stageRewrite.durationMs = Date.now() - rewriteStart;
      stageRewrite.detail = `${rewriteResult.rewrittenQueries.length} query variants generated`;
      trace.queryRewrite = rewriteResult;
    } catch (error) {
      stageRewrite.status = "failed";
      stageRewrite.detail = "Query rewrite failed — using original query";
      rewriteResult = {
        originalQuery: currentQuery,
        rewrittenQueries: [currentQuery],
        reasoning: "Fallback",
      };
    }

    // ── Stage 2: HyDE Generation ──────────────────────────────────
    const stageHyde = trace.stages[1];
    stageHyde.status = "running";
    const hydeStart = Date.now();

    let hydeResult: HyDEResult;
    try {
      hydeResult = await generateHyDE(currentQuery);
      stageHyde.status = "completed";
      stageHyde.durationMs = Date.now() - hydeStart;
      stageHyde.detail = `Hypothetical doc: ${hydeResult.hypotheticalDocument.substring(0, 80)}...`;
      trace.hyde = hydeResult;
    } catch (error) {
      stageHyde.status = "failed";
      stageHyde.detail = "HyDE generation failed — skipping";
      hydeResult = {
        originalQuery: currentQuery,
        hypotheticalDocument: currentQuery,
        embedding: [],
      };
    }

    // ── Stage 3: Sub-Query Decomposition ──────────────────────────
    const stageSubQuery = trace.stages[2];
    stageSubQuery.status = "running";
    const subQueryStart = Date.now();

    let subQueryResult: SubQueryResult;
    try {
      subQueryResult = await decomposeQuery(currentQuery);
      stageSubQuery.status = "completed";
      stageSubQuery.durationMs = Date.now() - subQueryStart;
      if (subQueryResult.needsDecomposition) {
        stageSubQuery.detail = `Decomposed into ${subQueryResult.subQueries.length} sub-queries`;
      } else {
        stageSubQuery.detail = "Simple query — no decomposition needed";
      }
      trace.subQueryDecomposition = subQueryResult;
    } catch (error) {
      stageSubQuery.status = "failed";
      stageSubQuery.detail = "Decomposition failed — using original query";
      subQueryResult = {
        originalQuery: currentQuery,
        needsDecomposition: false,
        subQueries: [currentQuery],
        reasoning: "Fallback",
      };
    }

    // ── Stage 4: Multi-Source Retrieval ────────────────────────────
    const stageRetrieval = trace.stages[3];
    stageRetrieval.status = "running";
    const retrievalStart = Date.now();

    // Combine all query variants: rewritten queries + sub-queries
    const allQueries = [
      ...new Set([
        currentQuery,
        ...rewriteResult.rewrittenQueries,
        ...(subQueryResult.needsDecomposition
          ? subQueryResult.subQueries
          : []),
      ]),
    ];

    let allDocs: Document[] = [];
    try {
      // Standard multi-query retrieval
      const multiDocs = await multiQueryRetrieve(
        allQueries,
        collectionName,
        5
      );
      allDocs.push(...multiDocs);

      // HyDE-based retrieval (if embedding was generated)
      if (hydeResult.embedding.length > 0) {
        const hydeDocs = await hydeRetrieve(
          hydeResult.embedding,
          collectionName,
          5
        );
        // Deduplicate against already-retrieved docs
        const existingHashes = new Set(
          allDocs.map((d) => d.pageContent.substring(0, 100))
        );
        for (const doc of hydeDocs) {
          if (!existingHashes.has(doc.pageContent.substring(0, 100))) {
            allDocs.push(doc);
          }
        }
      }

      stageRetrieval.status = "completed";
      stageRetrieval.durationMs = Date.now() - retrievalStart;
      stageRetrieval.detail = `${allDocs.length} unique chunks retrieved from ${allQueries.length} queries`;
      trace.retrievedCount = allDocs.length;
    } catch (error) {
      stageRetrieval.status = "failed";
      stageRetrieval.detail = "Retrieval failed";
      throw new Error("Retrieval failed — cannot generate answer without context");
    }

    // ── Stage 5: Cross-Encoder Reranking ──────────────────────────
    const stageRerank = trace.stages[4];
    stageRerank.status = "running";
    const rerankStart = Date.now();

    let rerankResult: RerankResult;
    try {
      rerankResult = await rerankDocuments(currentQuery, allDocs);
      stageRerank.status = "completed";
      stageRerank.durationMs = Date.now() - rerankStart;
      stageRerank.detail = `${rerankResult.rankedDocuments.length} chunks kept, ${rerankResult.removedCount} filtered out`;
      trace.rerank = {
        totalScored: rerankResult.totalScored,
        removedCount: rerankResult.removedCount,
        topScores: rerankResult.rankedDocuments
          .slice(0, 5)
          .map((d) => d.relevanceScore),
      };
    } catch (error) {
      stageRerank.status = "failed";
      stageRerank.detail = "Reranking failed — using unranked results";
      // Fallback: wrap all docs as RankedDocuments with neutral score
      rerankResult = {
        rankedDocuments: allDocs.map((doc) => ({
          document: doc,
          relevanceScore: 5,
          reasoning: "Reranking failed",
        })),
        removedCount: 0,
        totalScored: allDocs.length,
      };
    }

    // ── Stage 6: Token Management ─────────────────────────────────
    const stageTokens = trace.stages[5];
    stageTokens.status = "running";
    const tokensStart = Date.now();

    let tokenResult: TokenBudgetResult;
    try {
      tokenResult = fitToTokenBudget(rerankResult.rankedDocuments);
      stageTokens.status = "completed";
      stageTokens.durationMs = Date.now() - tokensStart;
      stageTokens.detail = `${tokenResult.selectedDocuments.length} chunks, ~${tokenResult.totalTokensUsed} tokens (budget: ${tokenResult.budgetLimit})`;
      trace.tokenBudget = {
        tokensUsed: tokenResult.totalTokensUsed,
        budgetLimit: tokenResult.budgetLimit,
        chunksUsed: tokenResult.selectedDocuments.length,
        chunksDropped: tokenResult.droppedCount,
      };
    } catch (error) {
      stageTokens.status = "failed";
      // Fallback: use raw context from reranked docs
      const fallbackContext = rerankResult.rankedDocuments
        .slice(0, 5)
        .map((rd, i) => `[${i + 1}] ${rd.document.pageContent}`)
        .join("\n\n");
      tokenResult = {
        selectedDocuments: rerankResult.rankedDocuments.slice(0, 5),
        droppedCount: 0,
        totalTokensUsed: 0,
        budgetLimit: 0,
        contextString: fallbackContext,
      };
    }

    contextForJudge = tokenResult.contextString;

    // ── Stage 7: Generation ───────────────────────────────────────
    const stageGeneration = trace.stages[6];
    stageGeneration.status = "running";
    const genStart = Date.now();

    const systemPrompt = `You are an expert AI assistant that answers questions based strictly on the provided document context.

Rules:
- ONLY use information from the provided context to answer
- If the context doesn't contain enough information, say: "I don't have enough information in the document to answer that."
- Do NOT use your general knowledge or hallucinate facts
- Cite relevant source/page numbers when possible using [Source, Page X] format
- Provide comprehensive, well-structured answers
- Use bullet points or numbered lists for complex answers

Context:
${tokenResult.contextString}`;

    try {
      const response = await openai.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: currentQuery },
        ],
        temperature: 0.1,
      });

      finalAnswer = response.choices[0].message.content || "";
      stageGeneration.status = "completed";
      stageGeneration.durationMs = Date.now() - genStart;
      stageGeneration.detail = `${finalAnswer.length} chars generated`;

      // Build sources with relevance scores
      finalSources = tokenResult.selectedDocuments.map((rd) => ({
        source: rd.document.metadata?.source || "document",
        page:
          rd.document.metadata?.page ||
          rd.document.metadata?.loc?.pageNumber ||
          "N/A",
        relevanceScore: rd.relevanceScore,
      }));
    } catch (error) {
      stageGeneration.status = "failed";
      stageGeneration.detail = "Generation failed";
      throw new Error("Answer generation failed");
    }

    // ── Stage 8: LLM Judge ────────────────────────────────────────
    const stageJudge = trace.stages[7];
    stageJudge.status = "running";
    const judgeStart = Date.now();

    let judgeResult: JudgeResult;
    try {
      judgeResult = await judgeAnswer(
        currentQuery,
        finalAnswer,
        contextForJudge
      );
      stageJudge.status = "completed";
      stageJudge.durationMs = Date.now() - judgeStart;
      stageJudge.detail = `Score: ${judgeResult.overallScore}/5 — ${judgeResult.passed ? "PASSED ✓" : "FAILED ✗"}`;
      trace.judge = judgeResult;
    } catch (error) {
      stageJudge.status = "failed";
      stageJudge.detail = "Judge failed — answer passed by default";
      judgeResult = {
        scores: { relevance: 3, faithfulness: 3, completeness: 3 },
        overallScore: 3.0,
        passed: true,
        feedback: "Judge unavailable",
        suggestions: "None",
      };
      trace.judge = judgeResult;
    }

    // If judge passed or we're out of retries, return the answer
    if (judgeResult.passed || attempt >= MAX_RETRIES) {
      return {
        answer: finalAnswer,
        sources: finalSources,
        trace,
        totalDurationMs: Date.now() - pipelineStart,
      };
    }

    // ── Corrective RAG: Retry with improved query ─────────────────
    console.log(
      `[Corrective RAG] Judge failed (score: ${judgeResult.overallScore}). Retrying with suggestions: ${judgeResult.suggestions}`
    );
    currentQuery = `${query} (Hint: ${judgeResult.suggestions})`;
  }

  // Should never reach here, but just in case
  return {
    answer: finalAnswer,
    sources: finalSources,
    trace,
    totalDurationMs: Date.now() - pipelineStart,
  };
}
