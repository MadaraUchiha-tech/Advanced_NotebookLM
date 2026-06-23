import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import { extractText } from "unpdf";
import OpenAI from "openai";

const QDRANT_URL = process.env.QDRANT_URL || "http://localhost:6333";
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;
const QDRANT_COLLECTION = process.env.QDRANT_COLLECTION || "rag-collection";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const OPENAI_BASE_URL =
  process.env.OPENAI_BASE_URL || "https://models.github.ai/inference";

export const embeddings = new OpenAIEmbeddings({
  modelName: "text-embedding-3-large",
  apiKey: OPENAI_API_KEY,
  configuration: { baseURL: OPENAI_BASE_URL },
});

export const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
  baseURL: OPENAI_BASE_URL,
});

/**
 * Chunking Strategy: Recursive Character Text Splitter
 *
 * Splits text hierarchically (paragraphs → sentences → words) to preserve
 * semantic coherence. Each chunk is enriched with positional metadata.
 *
 * Parameters tuned for advanced RAG:
 * - chunkSize: 800 chars (smaller for better precision after reranking)
 * - chunkOverlap: 150 chars (sufficient context bridging)
 */
export async function chunkDocuments(docs: Document[]): Promise<Document[]> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 800,
    chunkOverlap: 150,
  });

  const chunks = await splitter.splitDocuments(docs);

  // Enrich chunks with positional metadata
  return chunks.map((chunk, index) => {
    return new Document({
      pageContent: chunk.pageContent,
      metadata: {
        ...chunk.metadata,
        chunkIndex: index,
        totalChunks: chunks.length,
      },
    });
  });
}

export async function indexDocuments(
  docs: Document[],
  collectionName?: string
): Promise<QdrantVectorStore> {
  const chunks = await chunkDocuments(docs);

  const vectorStore = await QdrantVectorStore.fromDocuments(
    chunks,
    embeddings,
    {
      url: QDRANT_URL,
      apiKey: QDRANT_API_KEY,
      collectionName: collectionName || QDRANT_COLLECTION,
    }
  );

  return vectorStore;
}

export async function getVectorStore(
  collectionName?: string
): Promise<QdrantVectorStore> {
  const vectorStore = await QdrantVectorStore.fromExistingCollection(
    embeddings,
    {
      url: QDRANT_URL,
      apiKey: QDRANT_API_KEY,
      collectionName: collectionName || QDRANT_COLLECTION,
    }
  );

  return vectorStore;
}

/**
 * Multi-Query Retrieval
 *
 * Runs vector search for multiple query variations (from query rewriting
 * and sub-query decomposition) and merges + deduplicates the results.
 * This maximizes recall by capturing documents that match different
 * phrasings of the same intent.
 */
export async function multiQueryRetrieve(
  queries: string[],
  collectionName?: string,
  k: number = 5
): Promise<Document[]> {
  const vectorStore = await getVectorStore(collectionName);
  const retriever = vectorStore.asRetriever({ k });

  // Retrieve for each query variant in parallel
  const allResults = await Promise.all(
    queries.map((q) => retriever.invoke(q))
  );

  // Flatten and deduplicate by content hash
  const seen = new Set<string>();
  const deduplicated: Document[] = [];

  for (const docs of allResults) {
    for (const doc of docs) {
      const hash = simpleHash(doc.pageContent);
      if (!seen.has(hash)) {
        seen.add(hash);
        deduplicated.push(doc);
      }
    }
  }

  return deduplicated;
}

/**
 * HyDE-aware retrieval — searches using a pre-computed embedding vector
 * from the hypothetical document rather than raw query text.
 */
export async function hydeRetrieve(
  hydeEmbedding: number[],
  collectionName?: string,
  k: number = 5
): Promise<Document[]> {
  const vectorStore = await getVectorStore(collectionName);
  const results = await vectorStore.similaritySearchVectorWithScore(
    hydeEmbedding,
    k
  );
  return results.map(([doc]) => doc);
}

/**
 * Simple string hash for deduplication (FNV-1a inspired).
 */
function simpleHash(str: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = (hash * 0x01000193) >>> 0;
  }
  return hash.toString(36);
}

// ── File Processing ──────────────────────────────────────────────────

export async function processPdfFile(file: File): Promise<Document[]> {
  const bytes = await file.arrayBuffer();
  const buffer = new Uint8Array(bytes);

  const { text } = await extractText(buffer, { mergePages: false });

  return text.map(
    (pageText: string, index: number) =>
      new Document({
        pageContent: pageText.trim(),
        metadata: { page: index + 1 },
      })
  );
}

export async function processTextFile(file: File): Promise<Document[]> {
  const text = await file.text();
  const doc = new Document({
    pageContent: text,
    metadata: { source: file.name },
  });
  return [doc];
}
