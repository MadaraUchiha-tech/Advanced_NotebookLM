import { NextRequest, NextResponse } from "next/server";
import { processPdfFile, processTextFile, indexDocuments } from "@/lib/rag";
import { v4 as uuidv4 } from "uuid";
import { Document } from "@langchain/core/documents";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File | null;

    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    const fileType = file.type;
    const fileName = file.name;

    let docs;
    if (fileType === "application/pdf" || fileName.toLowerCase().endsWith(".pdf")) {
      docs = await processPdfFile(file);
    } else if (
      fileType === "text/plain" ||
      fileName.toLowerCase().endsWith(".txt") ||
      fileName.toLowerCase().endsWith(".md")
    ) {
      docs = await processTextFile(file);
    } else {
      return NextResponse.json(
        { error: "Unsupported file type. Only PDF and plain text files are supported." },
        { status: 400 }
      );
    }

    // Attach filename metadata
    docs = docs.map(
      (doc) =>
        new Document({
          pageContent: doc.pageContent,
          metadata: { ...doc.metadata, source: fileName },
        })
    );

    const collectionName = uuidv4();
    await indexDocuments(docs, collectionName);

    return NextResponse.json({
      success: true,
      message: `Document "${fileName}" indexed successfully with ${docs.length} pages.`,
      collectionName,
      pages: docs.length,
    });
  } catch (error: any) {
    console.error("Upload error:", error);
    return NextResponse.json(
      { error: error.message || "Failed to process document" },
      { status: 500 }
    );
  }
}
