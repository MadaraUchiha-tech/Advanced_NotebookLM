# Advanced NotebookLM RAG

A production-grade, advanced Retrieval-Augmented Generation (RAG) application. Users can upload any document (PDF or plain text) and have an intelligent conversation with it. Unlike a naive RAG system, this application implements cutting-edge techniques to maximize retrieval accuracy, prevent hallucinations, and evaluate its own answers.

## Advanced Architecture & Features

This project moves beyond standard semantic search by implementing a sophisticated multi-stage pipeline:

### 1. Query Rewriting & Translation
User queries are often ambiguous or lack context. The system uses an LLM to translate and rewrite raw user queries into clearer, highly optimized queries tailored for vector database retrieval.

### 2. Sub-Query Decomposition
Complex questions that require synthesizing multiple pieces of information are broken down into simpler, parallel sub-queries. The system retrieves documents for each sub-query independently and then merges and deduplicates the results.

### 3. HyDE (Hypothetical Document Embeddings)
Instead of embedding the user's raw query to find similar vectors, the system first generates a *hypothetical document* (an educated guess at the answer). It then embeds this hypothetical document to perform the similarity search, significantly improving retrieval accuracy by searching in the document answer space.

### 4. Cross-Encoder Re-Ranking
Standard vector similarity search can retrieve context that is topically similar but practically irrelevant. The system employs an LLM-based Cross-Encoder re-ranker to score and filter retrieved chunks strictly based on their direct relevance to the actual query.

### 5. Corrective RAG
If all retrieved chunks fall below a strict relevance threshold during the re-ranking phase, the system triggers a Corrective RAG pattern. It refuses to answer and informs the user that the document doesn't contain the requested information, effectively neutralizing out-of-context hallucinations.

### 6. Token & Context Window Management
Assembled contexts are rigorously monitored using a Token Manager to ensure they do not exceed the LLM's context window budget. The system safely truncates excess context and always prioritizes the highest-scoring chunks from the re-ranker to maximize prompt value.

### 7. LLM-as-a-Judge & Self-Correction
Before returning the final answer, an independent LLM judge evaluates the response across four specific metrics:
- **Faithfulness**: Is the answer grounded purely in the retrieved context?
- **Relevance**: Does it directly answer the user's original query?
- **Completeness**: Is the answer comprehensive?
- **Coherence**: Is the text well-structured and logical?

If the generated answer fails the evaluation threshold, the system feeds the judge's exact feedback back into the generation model and retries automatically to correct itself.

## Pipeline Modes

The application orchestrator dynamically adjusts the pipeline complexity based on the requested mode, allowing users to trade off processing speed for maximum accuracy:

| Mode | Rewrite | SubQuery | HyDE | Rerank | Judge | Top-K | Context Limit |
|------|---------|----------|------|--------|-------|-------|---------------|
| **Fast** | ❌ | ❌ | ❌ | Lightweight | ❌ | 3 | 2000 tokens |
| **Balanced** | ✅ | ❌ | ✅ | LLM-based | ❌ | 5 | 4000 tokens |
| **Accurate** | ✅ | ✅ | ✅ | LLM-based | ✅ | 10 | 8000 tokens |

## Tech Stack

- Next.js 16 (App Router)
- React 19 + TypeScript
- Tailwind CSS
- LangChain (`@langchain/openai`, `@langchain/qdrant`, `@langchain/textsplitters`)
- Qdrant Vector Database
- GitHub Models / Azure AI Inference API (Embeddings + LLMs)

## Environment Variables

Copy `env.example` to `.env.local` and fill in your keys:

```bash
OPENAI_API_KEY=ghp_your_github_token_here
OPENAI_BASE_URL=https://models.inference.ai.azure.com
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=rag-collection
```

**Getting a GitHub token:** Go to [GitHub Settings > Tokens](https://github.com/settings/tokens) and generate a classic token. No special scopes are required for GitHub Models.

## Local Development

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## API Endpoints

### Upload an Index Document
```bash
curl -X POST http://localhost:3000/api/upload \
  -F "file=@document.pdf"
```

### Query the RAG Pipeline
You can specify the pipeline mode (`fast`, `balanced`, or `accurate`) in the payload.
```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "collectionName": "...", "mode": "accurate"}'
```

The response includes the generated answer, the sources, and highly detailed diagnostics of every pipeline stage (time taken, judge scores, token utilization, skipped stages, corrective triggers, etc.).
