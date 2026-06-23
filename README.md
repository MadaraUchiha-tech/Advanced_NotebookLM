# NotebookLM RAG

A RAG-powered application where a user can upload any document (PDF or plain text) and have a conversation with it. The system processes the file, stores it intelligently, and answers natural language questions grounded in the document's actual content.

## Live Project

Deployed and accessible without any local setup.

## Features

- Upload PDF or plain text documents
- Automatic chunking, embedding, and indexing into a vector database
- Ask natural language questions about the document
- Answers grounded in the document -- not from the LLM's general knowledge
- Clean web UI for uploading and chatting

## RAG Pipeline

The full pipeline is implemented end-to-end:

1. **Ingestion** -- Accepts PDF and text files via file upload
2. **Chunking** -- Recursive Character Text Splitter
3. **Embedding** -- OpenAI `text-embedding-3-large`
4. **Storage** -- Qdrant vector database
5. **Retrieval** -- Similarity search to find top-k relevant chunks
6. **Generation** -- GitHub Models `gpt-4o-mini` with a strict system prompt that only uses retrieved context

## Chunking Strategy

**Recursive Character Text Splitter**

This strategy recursively splits text by characters, starting with larger separators (like double newlines for paragraphs) and moving to smaller ones (like single newlines for sentences, then spaces for words). It preserves semantic coherence better than fixed-size chunking alone.

| Parameter | Value |
|-----------|-------|
| chunkSize | 1000 characters |
| chunkOverlap | 200 characters |

## Tech Stack

- Next.js 16 (App Router)
- React 19 + TypeScript
- Tailwind CSS
- LangChain (`@langchain/openai`, `@langchain/qdrant`, `@langchain/textsplitters`)
- Qdrant vector database
- GitHub Models / Azure AI Inference API (embeddings + LLM)

## Environment Variables

Copy `env.example` to `.env.local` and fill in your keys:

```
OPENAI_API_KEY=ghp_your_github_token_here
OPENAI_BASE_URL=https://models.inference.ai.azure.com
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=rag-collection
```

**Getting a GitHub token:** Go to [GitHub Settings > Tokens](https://github.com/settings/tokens) and generate a classic token with no special scopes required.

## Local Development

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload and index a document (PDF or text) |
| `/api/query` | POST | Ask a question about the indexed document |

### Upload

```bash
curl -X POST http://localhost:3000/api/upload \
  -F "file=@document.pdf"
```

Response:
```json
{
  "success": true,
  "message": "Document \"document.pdf\" indexed successfully with 10 pages.",
  "collectionName": "...",
  "pages": 10
}
```

### Query

```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "collectionName": "..."}'
```

Response:
```json
{
  "success": true,
  "answer": "...",
  "sources": ["document.pdf (page 3)", "document.pdf (page 5)"]
}
```
# RAG_system
