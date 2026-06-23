"use client";

import { useState, useRef, useEffect } from "react";

// ── Types ────────────────────────────────────────────────────────────

interface PipelineStageStatus {
  name: string;
  status: "pending" | "running" | "completed" | "skipped" | "failed";
  durationMs?: number;
  detail?: string;
}

interface JudgeScores {
  relevance: number;
  faithfulness: number;
  completeness: number;
}

interface PipelineTrace {
  stages: PipelineStageStatus[];
  queryRewrite?: {
    originalQuery: string;
    rewrittenQueries: string[];
    reasoning: string;
  };
  hyde?: {
    hypotheticalDocument: string;
  };
  subQueryDecomposition?: {
    needsDecomposition: boolean;
    subQueries: string[];
    reasoning: string;
  };
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
  judge?: {
    scores: JudgeScores;
    overallScore: number;
    passed: boolean;
    feedback: string;
  };
  retryCount: number;
}

interface Source {
  source: string;
  page: string | number;
  relevanceScore: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  trace?: PipelineTrace;
  totalDurationMs?: number;
}

// ── Stage Icon Component ─────────────────────────────────────────────

function StageIcon({ status }: { status: string }) {
  switch (status) {
    case "completed":
      return (
        <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
        </svg>
      );
    case "running":
      return (
        <div className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
      );
    case "failed":
      return (
        <svg className="w-4 h-4 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      );
    case "skipped":
      return (
        <svg className="w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M13 5l7 7-7 7M5 5l7 7-7 7" />
        </svg>
      );
    default:
      return <div className="w-4 h-4 rounded-full border-2 border-slate-600" />;
  }
}

// ── Score Badge Component ────────────────────────────────────────────

function ScoreBadge({ label, score, max = 5 }: { label: string; score: number; max?: number }) {
  const ratio = score / max;
  const cls = ratio >= 0.7 ? "score-high" : ratio >= 0.5 ? "score-medium" : "score-low";
  return (
    <span className={`score-badge ${cls}`}>
      {label}: {score}/{max}
    </span>
  );
}

// ── Pipeline Trace Panel ─────────────────────────────────────────────

function PipelinePanel({ trace, totalDurationMs }: { trace: PipelineTrace; totalDurationMs?: number }) {
  const [expanded, setExpanded] = useState(false);
  const completedStages = trace.stages.filter((s) => s.status === "completed").length;
  const progress = (completedStages / trace.stages.length) * 100;

  return (
    <div className="glass-card mt-3 overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-slate-800/50 transition-colors cursor-pointer"
        id="pipeline-toggle"
      >
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
          </svg>
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Advanced Pipeline
          </span>
          {trace.retryCount > 0 && (
            <span className="text-xs bg-amber-500/20 text-amber-400 px-1.5 py-0.5 rounded-full border border-amber-500/30">
              Retry #{trace.retryCount}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {totalDurationMs && (
            <span className="text-xs text-slate-500">{(totalDurationMs / 1000).toFixed(1)}s</span>
          )}
          <span className="text-xs text-slate-400">{completedStages}/{trace.stages.length}</span>
          <svg
            className={`w-4 h-4 text-slate-500 transition-transform ${expanded ? "rotate-180" : ""}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Progress bar */}
      <div className="pipeline-progress mx-4">
        <div className="pipeline-progress-fill" style={{ width: `${progress}%` }} />
      </div>

      {/* Expanded details */}
      {expanded && (
        <div className="px-4 pt-3 pb-4 space-y-2 stage-enter">
          {trace.stages.map((stage, idx) => (
            <div
              key={idx}
              className={`flex items-start gap-3 py-1.5 ${stage.status === "running" ? "stage-running rounded-lg px-2 -mx-2" : ""}`}
            >
              <div className="mt-0.5">
                <StageIcon status={stage.status} />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-medium ${stage.status === "completed" ? "text-slate-200" : stage.status === "running" ? "text-indigo-300" : stage.status === "failed" ? "text-red-400" : "text-slate-500"}`}>
                    {stage.name}
                  </span>
                  {stage.durationMs !== undefined && (
                    <span className="text-[0.65rem] text-slate-600">
                      {stage.durationMs}ms
                    </span>
                  )}
                </div>
                {stage.detail && (
                  <p className="text-[0.65rem] text-slate-500 mt-0.5 truncate">
                    {stage.detail}
                  </p>
                )}
              </div>
            </div>
          ))}

          {/* Judge Scores */}
          {trace.judge && (
            <div className="pt-2 mt-2 border-t border-slate-800">
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-xs font-semibold text-slate-400">Quality Assessment</span>
                <span className={`text-xs px-1.5 py-0.5 rounded-full font-semibold ${trace.judge.passed ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30" : "bg-red-500/20 text-red-400 border border-red-500/30"}`}>
                  {trace.judge.passed ? "PASSED" : "FAILED"}
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                <ScoreBadge label="Relevance" score={trace.judge.scores.relevance} />
                <ScoreBadge label="Faithful" score={trace.judge.scores.faithfulness} />
                <ScoreBadge label="Complete" score={trace.judge.scores.completeness} />
              </div>
              <p className="text-[0.65rem] text-slate-500 mt-1.5 leading-relaxed">
                {trace.judge.feedback}
              </p>
            </div>
          )}

          {/* Query Rewrite Detail */}
          {trace.queryRewrite && trace.queryRewrite.rewrittenQueries.length > 1 && (
            <div className="pt-2 mt-2 border-t border-slate-800">
              <span className="text-[0.65rem] font-semibold text-slate-400 block mb-1">
                Query Variants
              </span>
              {trace.queryRewrite.rewrittenQueries.map((q, i) => (
                <p key={i} className="text-[0.65rem] text-slate-500 pl-2 border-l border-indigo-500/30 mb-1">
                  {q}
                </p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main Page Component ──────────────────────────────────────────────

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [collectionName, setCollectionName] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setUploadMessage("");
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setUploadMessage("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (data.success) {
        setCollectionName(data.collectionName);
        setUploadMessage(data.message);
        setMessages([]);
      } else {
        setUploadMessage(data.error || "Upload failed");
      }
    } catch (err: any) {
      setUploadMessage(err.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    if (!collectionName) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Please upload a document first using the sidebar before asking questions.",
        },
      ]);
      setQuery("");
      return;
    }

    const userMessage: Message = { role: "user", content: query };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    setQuery("");

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMessage.content, collectionName }),
      });
      const data = await res.json();

      if (data.success) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.answer,
            sources: data.sources,
            trace: data.trace,
            totalDurationMs: data.totalDurationMs,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${data.error}` },
        ]);
      }
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${err.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen" style={{ background: "var(--background)" }}>
      {/* ── Sidebar ──────────────────────────────────────────────── */}
      <aside className="w-80 flex flex-col p-5 border-r" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
        {/* Logo */}
        <div className="mb-6">
          <div className="flex items-center gap-2.5 mb-1">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: "linear-gradient(135deg, var(--accent), #4f46e5)" }}>
              <svg className="w-4.5 h-4.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <div>
              <h1 className="text-base font-bold" style={{ color: "var(--text-primary)" }}>
                Advanced RAG
              </h1>
            </div>
          </div>
        </div>

        {/* Upload Section */}
        <div className="glass-card p-4 mb-4">
          <label className="block text-xs font-semibold mb-2 uppercase tracking-wider" style={{ color: "var(--text-secondary)" }}>
            Upload Document
          </label>
          <input
            type="file"
            accept=".pdf,.txt,.md"
            onChange={handleFileChange}
            id="file-upload"
            className="block w-full text-xs mb-3 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:cursor-pointer"
            style={{
              color: "var(--text-muted)",
            }}
          />
          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            id="upload-button"
            className="w-full py-2 px-4 rounded-lg text-xs font-semibold text-white transition-all disabled:opacity-40 disabled:cursor-not-allowed hover:brightness-110 cursor-pointer"
            style={{ background: "linear-gradient(135deg, var(--accent), #4f46e5)" }}
          >
            {uploading ? (
              <span className="flex items-center justify-center gap-2">
                <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Processing...
              </span>
            ) : (
              "Upload & Index"
            )}
          </button>
        </div>

        {/* Upload Status */}
        {uploadMessage && (
          <div
            className={`p-3 rounded-lg text-xs mb-4 ${uploadMessage.includes("failed") || uploadMessage.includes("Error")
              ? "bg-red-500/10 text-red-400 border border-red-500/20"
              : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
              }`}
          >
            {uploadMessage}
          </div>
        )}

        {collectionName && (
          <div className="glass-card p-3 mb-4">
            <p className="text-[0.65rem] font-semibold uppercase tracking-wider" style={{ color: "var(--accent-light)" }}>
              Active Collection
            </p>
            <p className="text-[0.6rem] break-all mt-1" style={{ color: "var(--text-muted)" }}>
              {collectionName}
            </p>
          </div>
        )}

        {/* Pipeline Info */}
      </aside>

      {/* ── Main Chat Area ───────────────────────────────────────── */}
      <main className="flex-1 flex flex-col" style={{ background: "var(--background)" }}>
        <div className="flex-1 overflow-y-auto p-6">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-center">
              <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center mb-5"
                style={{ background: "var(--accent-glow)", border: "1px solid rgba(99, 102, 241, 0.2)" }}
              >
                <svg className="w-8 h-8" style={{ color: "var(--accent-light)" }} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h2 className="text-lg font-semibold mb-1.5" style={{ color: "var(--text-primary)" }}>
                Advanced RAG Pipeline
              </h2>
              <p className="text-sm max-w-md leading-relaxed" style={{ color: "var(--text-muted)" }}>
                Upload a document and ask questions.
              </p>
            </div>
          )}

          <div className="max-w-3xl mx-auto space-y-5">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div className={`max-w-[85%] ${msg.role === "user" ? "message-user" : "message-assistant"} px-5 py-3`}>
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">
                    {msg.content}
                  </p>

                  {/* Sources with Relevance Scores */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-3 pt-3" style={{ borderTop: "1px solid var(--border)" }}>
                      <p className="text-[0.65rem] font-semibold mb-1.5 uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
                        Sources
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {msg.sources.map((src, sIdx) => (
                          <span
                            key={sIdx}
                            className="text-[0.65rem] px-2 py-1 rounded-md inline-flex items-center gap-1"
                            style={{ background: "var(--background)", color: "var(--text-secondary)" }}
                          >
                            {src.source} (p.{src.page})
                            {src.relevanceScore && (
                              <span className={`text-[0.6rem] font-semibold ${src.relevanceScore >= 7 ? "text-emerald-400" : src.relevanceScore >= 5 ? "text-amber-400" : "text-red-400"
                                }`}>
                                {src.relevanceScore.toFixed(0)}
                              </span>
                            )}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Pipeline Trace Panel */}
                  {msg.trace && (
                    <PipelinePanel trace={msg.trace} totalDurationMs={msg.totalDurationMs} />
                  )}
                </div>
              </div>
            ))}

            {/* Loading Indicator */}
            {loading && (
              <div className="flex justify-start">
                <div className="message-assistant px-5 py-4">
                  <div className="flex items-center gap-3">
                    <div className="flex space-x-1.5">
                      <div className="w-2 h-2 rounded-full animate-bounce" style={{ background: "var(--accent)", animationDelay: "0ms" }} />
                      <div className="w-2 h-2 rounded-full animate-bounce" style={{ background: "var(--accent-light)", animationDelay: "150ms" }} />
                      <div className="w-2 h-2 rounded-full animate-bounce" style={{ background: "var(--accent)", animationDelay: "300ms" }} />
                    </div>
                    <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                      Running advanced pipeline...
                    </span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* ── Input Bar ──────────────────────────────────────────── */}
        <div className="p-4" style={{ borderTop: "1px solid var(--border)", background: "var(--surface)" }}>
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex gap-3">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question about your document..."
              disabled={loading}
              id="query-input"
              className="flex-1 px-4 py-3 rounded-xl text-sm focus:outline-none focus:ring-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              style={{
                background: "var(--background)",
                border: "1px solid var(--border)",
                color: "var(--text-primary)",
                "--tw-ring-color": "var(--accent)",
              } as React.CSSProperties}
            />
            <button
              type="submit"
              disabled={!query.trim() || loading}
              id="send-button"
              className="px-6 py-3 rounded-xl text-sm font-semibold text-white transition-all disabled:opacity-40 disabled:cursor-not-allowed hover:brightness-110 cursor-pointer"
              style={{ background: "linear-gradient(135deg, var(--accent), #4f46e5)" }}
            >
              Send
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}
