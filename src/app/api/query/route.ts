import { NextRequest, NextResponse } from "next/server";
import { runAdvancedPipeline } from "@/lib/pipeline";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { query, collectionName } = body;

    if (!query || typeof query !== "string") {
      return NextResponse.json({ error: "Query is required" }, { status: 400 });
    }

    if (!collectionName || typeof collectionName !== "string") {
      return NextResponse.json(
        {
          error:
            "Collection name is required. Please upload a document first.",
        },
        { status: 400 }
      );
    }

    const result = await runAdvancedPipeline(query, collectionName);

    return NextResponse.json({
      success: true,
      answer: result.answer,
      sources: result.sources,
      trace: result.trace,
      totalDurationMs: result.totalDurationMs,
    });
  } catch (error: any) {
    console.error("Query error:", error);
    return NextResponse.json(
      { error: error.message || "Failed to generate answer" },
      { status: 500 }
    );
  }
}
