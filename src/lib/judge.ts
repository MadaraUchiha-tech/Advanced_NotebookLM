import OpenAI from "openai";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const OPENAI_BASE_URL =
  process.env.OPENAI_BASE_URL || "https://models.github.ai/inference";

const openai = new OpenAI({ apiKey: OPENAI_API_KEY, baseURL: OPENAI_BASE_URL });

export interface JudgeScores {
  relevance: number; // 1-5: Does the answer address the question?
  faithfulness: number; // 1-5: Is the answer grounded in the provided context?
  completeness: number; // 1-5: Does the answer cover all aspects of the question?
}

export interface JudgeResult {
  scores: JudgeScores;
  overallScore: number; // Average of all scores
  passed: boolean; // True if overallScore >= threshold
  feedback: string; // Explanation of the evaluation
  suggestions: string; // How to improve (used for corrective rewrite)
}

const PASS_THRESHOLD = 3.0; // Out of 5 — answers below this trigger a retry

/**
 * LLM Judge — Stage 6 of the Advanced RAG Pipeline
 *
 * Evaluates the quality of a generated answer on three axes:
 *
 * 1. **Relevance**: Does the answer actually address the user's question?
 *    - Low score → query rewrite likely needed
 *
 * 2. **Faithfulness**: Is every claim in the answer supported by the context?
 *    - Low score → hallucination detected
 *
 * 3. **Completeness**: Does the answer cover all aspects of the question?
 *    - Low score → sub-query decomposition or more retrieval needed
 *
 * If the overall score falls below the threshold, the pipeline retries with
 * a corrected query (corrective RAG pattern).
 */
export async function judgeAnswer(
  query: string,
  answer: string,
  context: string
): Promise<JudgeResult> {
  const systemPrompt = `You are an expert answer quality evaluator for a Retrieval-Augmented Generation system.

Evaluate the given answer against the user's query and the retrieved context.

Score each dimension from 1-5:

**Relevance** (Does the answer address the question?):
1: Completely off-topic
2: Tangentially related but misses the point
3: Partially addresses the question
4: Mostly addresses the question with minor gaps
5: Directly and fully addresses the question

**Faithfulness** (Is the answer grounded in the context?):
1: Contains major unsupported claims or fabrications
2: Several claims not supported by context
3: Mostly supported but some unverified claims
4: Nearly all claims traceable to context
5: Every claim is directly supported by context

**Completeness** (Does the answer cover all aspects?):
1: Addresses almost none of the question's scope
2: Covers less than half of what's asked
3: Covers the main point but misses secondary aspects
4: Covers most aspects with minor omissions
5: Comprehensively covers all aspects of the question

Respond in valid JSON:
{
  "relevance": <1-5>,
  "faithfulness": <1-5>,
  "completeness": <1-5>,
  "feedback": "2-3 sentence evaluation summary",
  "suggestions": "Specific suggestion for improvement if scores are low, or 'None' if quality is good"
}`;

  const userPrompt = `**User Query:** ${query}

**Retrieved Context:**
${context.substring(0, 3000)}

**Generated Answer:**
${answer}`;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: 0.0,
      response_format: { type: "json_object" },
    });

    const content = response.choices[0].message.content || "{}";
    const parsed = JSON.parse(content);

    const scores: JudgeScores = {
      relevance: clampScore(parsed.relevance),
      faithfulness: clampScore(parsed.faithfulness),
      completeness: clampScore(parsed.completeness),
    };

    const overallScore =
      (scores.relevance + scores.faithfulness + scores.completeness) / 3;

    return {
      scores,
      overallScore: Math.round(overallScore * 10) / 10,
      passed: overallScore >= PASS_THRESHOLD,
      feedback: parsed.feedback || "No feedback provided",
      suggestions: parsed.suggestions || "None",
    };
  } catch (error) {
    console.error("Judge evaluation failed:", error);
    // On failure, pass by default (don't block the user)
    return {
      scores: { relevance: 3, faithfulness: 3, completeness: 3 },
      overallScore: 3.0,
      passed: true,
      feedback: "Judge evaluation failed — answer passed by default",
      suggestions: "None",
    };
  }
}

function clampScore(score: number | undefined): number {
  if (score === undefined || score === null) return 3;
  return Math.min(5, Math.max(1, Math.round(score)));
}
