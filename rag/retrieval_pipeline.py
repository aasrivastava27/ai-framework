import asyncio
import json
import os
import faiss
from openai import OpenAI, BadRequestError
from dotenv import load_dotenv

from mcp_client import MCPToolBox

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode

import pickle

with open("nodes.pkl", "rb") as f:
    all_nodes = pickle.load(f)

from rank_bm25 import BM25Okapi
tokenized_docs = []

for node in all_nodes:
    tokenized_docs.append(node.text.lower().split())

bm25 = BM25Okapi(tokenized_docs)
import cohere

# ---------------------------------------
# Load environment variables
# ---------------------------------------

load_dotenv()

# ---------------------------------------
# Initialize cohere client
# ---------------------------------------
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# ---------------------------------------
# Configure embedding model
# ---------------------------------------

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# ---------------------------------------
# Initialize GROQ client
# ---------------------------------------

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ---------------------------------------
# Load FAISS index
# ---------------------------------------

faiss_index = faiss.read_index("faiss.index")

vector_store = FaissVectorStore(
    faiss_index=faiss_index
)

storage_context = StorageContext.from_defaults(
    persist_dir="storage",
    vector_store=vector_store
)

index = load_index_from_storage(storage_context)

# ---------------------------------------
# Create Retriever
# ---------------------------------------

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
    vector_store_query_mode=VectorStoreQueryMode.MMR,
    mmr_threshold=0.5
)

# ---------------------------------------
# List available source files (so tool calls
# don't have to guess filenames)
# ---------------------------------------

DOCS_DIR = "docs"
available_files = sorted(os.listdir(DOCS_DIR))

MAX_TOOL_ROUNDS = 5


async def main():

    # ---------------------------------------
    # Connect to MCP servers
    # ---------------------------------------

    mcp_tools = MCPToolBox()
    await mcp_tools.connect("mcp_config.json")

    # ---------------------------------------
    # Chat History
    # ---------------------------------------

    chat_history = []

    try:
        while True:

            query = input("\nQuestion (type 'quit' to exit): ")

            if query.lower() == "quit":
                break

            # -------------------------------------------------
            # Rewrite follow-up question into standalone query
            # -------------------------------------------------

            rewrite_prompt = f"""
You are a query rewriting assistant.

Conversation history:
{chat_history}

Latest user question:
{query}

Rewrite the latest question into a standalone question by resolving any references
such as "it", "they", "that company", etc.

If the latest question is already standalone, return it unchanged.

Return ONLY the rewritten question.
"""

            rewrite_response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                temperature=0,
                max_tokens=100,
                messages=[
                    {
                        "role": "user",
                        "content": rewrite_prompt
                    }
                ]
            )

            revised_query = rewrite_response.choices[0].message.content.strip()

            print("\nStandalone Query:")
            print(revised_query)

            # -------------------------------------------------
            # Generate Multiple Query Variations
            # -------------------------------------------------

            multi_query_prompt = f"""
Generate 3 different search queries for the following question.

The alternative queries should retrieve complementary information while preserving
the original meaning.

Original Query:
{revised_query}

Return ONLY the queries, one per line.
"""

            multi_query_response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                temperature=0,
                max_tokens=150,
                messages=[
                    {
                        "role": "user",
                        "content": multi_query_prompt
                    }
                ]
            )

            query_variations = [revised_query]

            lines = multi_query_response.choices[0].message.content.split("\n")

            for q in lines:
                if q.strip():
                    cleaned_query = q.strip("-• ").strip()
                    query_variations.append(cleaned_query)

            print("\nGenerated Queries")
            print("-" * 70)

            for i, q in enumerate(query_variations, 1):
                print(f"{i}. {q}")

            # -------------------------------------------------
            # Retrieve chunks for every query
            # -------------------------------------------------

            all_contexts = []
            for q in query_variations:

                # -------------------------
                # FAISS Retrieval
                # -------------------------

                nodes = retriever.retrieve(q)

                # -------------------------
                # BM25 Retrieval
                # -------------------------

                tokenized_query = q.lower().split()

                scores = bm25.get_scores(tokenized_query)

                # Find top 3 BM25 chunks

                top_indices = []

                temp_scores = scores.copy()

                for _ in range(5):

                    highest_score = -1
                    highest_index = -1

                    for i in range(len(temp_scores)):

                        if temp_scores[i] > highest_score:
                            highest_score = temp_scores[i]
                            highest_index = i

                    top_indices.append(highest_index)

                    temp_scores[highest_index] = -1

                bm25_nodes = []

                for idx in top_indices:
                    bm25_nodes.append(all_nodes[idx])

                # -------------------------
                # Store both FAISS and BM25 chunks
                # -------------------------

                query_context = []

                # FAISS chunks
                for node in nodes:
                    query_context.append(node.text)

                # BM25 chunks
                for node in bm25_nodes:
                    query_context.append(node.text)

                all_contexts.append(query_context)

            # -------------------------------------------------
            # Reciprocal Rank Fusion
            # -------------------------------------------------

            rrf_scores = {}
            unique_chunks = []

            k = 60

            # Calculate RRF scores
            for query_context in all_contexts:

                rank = 1

                for chunk in query_context:

                    # Store chunk only once
                    if chunk not in unique_chunks:
                        unique_chunks.append(chunk)

                    # Initialize score if first time seen
                    if chunk not in rrf_scores:
                        rrf_scores[chunk] = 0

                    # Add RRF score
                    rrf_scores[chunk] = rrf_scores[chunk] + (1 / (k + rank))

                    rank = rank + 1

            # Sort chunks according to RRF score (highest first)

            fused_chunks = unique_chunks.copy()

            for i in range(len(fused_chunks)):
                best = i

                for j in range(i + 1, len(fused_chunks)):
                    if rrf_scores[fused_chunks[j]] > rrf_scores[fused_chunks[best]]:
                        best = j

                fused_chunks[i], fused_chunks[best] = fused_chunks[best], fused_chunks[i]

            print("\nChunks After Reciprocal Rank Fusion")
            print("-" * 70)

            for i in range(len(fused_chunks)):
                chunk = fused_chunks[i]

                print("\nChunk", i + 1)
                print("Score:", round(rrf_scores[chunk], 5))
                print(chunk[:250] + "...")

            # -------------------------------------------------
            # Cohere Reranking
            # -------------------------------------------------

            rerank_response = co.rerank(
                model="rerank-v3.5",
                query=revised_query,
                documents=fused_chunks,
                top_n=5
            )

            reranked_chunks = []

            print("\nChunks After Cohere Reranking")
            print("-" * 70)

            for result in rerank_response.results:

                chunk = fused_chunks[result.index]
                reranked_chunks.append(chunk)

                print(f"\nRelevance Score: {result.relevance_score:.4f}")
                print(chunk[:250] + "...")

            # -------------------------------------------------
            # Build Prompt
            # -------------------------------------------------

            prompt = f"""
You are an intelligent AI assistant.

Primarily use the retrieved contexts below to answer the user's question.

The contexts are grouped by search query.
Different groups may contain overlapping or complementary information.

If the retrieved contexts are not enough to answer the question, you may use
the available tools to look up additional information (for example, reading
a raw source file). Only do this when the retrieved contexts are insufficient.

When calling a file tool, you MUST use one of these exact filenames (do not
guess or invent a filename):
{available_files}

If the answer still cannot be found, reply:

"I could not find the answer in the provided documents."

Retrieved Contexts:
{reranked_chunks}

Original User Question:
{query}
"""

            # -------------------------------------------------
            # Generate Final Answer (agentic tool-calling loop)
            # -------------------------------------------------

            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            answer = "I could not produce a final answer within the tool-call limit."

            for _ in range(MAX_TOOL_ROUNDS):

                try:
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        max_tokens=500,
                        temperature=0,
                        messages=messages,
                        tools=mcp_tools.openai_tools or None
                    )
                except BadRequestError as e:
                    print(f"\n[MCP] Tool call generation failed ({e}); "
                          f"answering from retrieved context only.")

                    fallback_response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        max_tokens=500,
                        temperature=0,
                        messages=[{"role": "user", "content": prompt}]
                    )

                    answer = fallback_response.choices[0].message.content
                    break

                response_message = response.choices[0].message

                if not response_message.tool_calls:
                    answer = response_message.content
                    break

                messages.append({
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [
                        tool_call.model_dump()
                        for tool_call in response_message.tool_calls
                    ]
                })

                for tool_call in response_message.tool_calls:

                    try:
                        tool_args = json.loads(tool_call.function.arguments)

                        print(f"\n[MCP] Calling tool: {tool_call.function.name}({tool_args})")

                        tool_result = await mcp_tools.call_tool(
                            tool_call.function.name,
                            tool_args
                        )
                    except Exception as e:
                        tool_result = f"Tool call failed: {e}"
                        print(f"\n[MCP] {tool_result}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

            # -------------------------------------------------
            # Save Chat History
            # -------------------------------------------------

            chat_history.append({
                "question": query,
                "answer": answer
            })

            # -------------------------------------------------
            # Print Final Answer
            # -------------------------------------------------

            print("\n")
            print("=" * 70)
            print("Final Answer")
            print("=" * 70)
            print(answer)

    finally:
        await mcp_tools.close()


if __name__ == "__main__":
    asyncio.run(main())
