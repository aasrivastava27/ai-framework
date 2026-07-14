import os
import faiss
import pickle
import cohere

from dotenv import load_dotenv

load_dotenv("config/.env")
from openai import OpenAI

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from rank_bm25 import BM25Okapi


class RAG:
    """
    Complete Retrieval-Augmented Generation pipeline.

    Public API:

        rag = RAG()

        answer = rag.answer(
            question,
            chat_history
        )
    """

    def __init__(self):

        self.client = None
        self.co = None
        self.retriever = None
        self.bm25 = None
        self.all_nodes = None

        self.model = "openai/gpt-oss-120b"

        self.load_models()

    # -------------------------------------------------
    # Load Models
    # -------------------------------------------------

    def load_models(self):

        load_dotenv()

        # ---------------------------------------
        # Load Nodes
        # ---------------------------------------

        with open("rag/vectorstore/nodes.pkl", "rb") as f:
            self.all_nodes = pickle.load(f)

        # ---------------------------------------
        # BM25
        # ---------------------------------------

        tokenized_docs = []

        for node in self.all_nodes:
            tokenized_docs.append(
                node.text.lower().split()
            )

        self.bm25 = BM25Okapi(tokenized_docs)

        # ---------------------------------------
        # Cohere
        # ---------------------------------------

        self.co = cohere.Client(
            os.getenv("COHERE_API_KEY")
        )

        # ---------------------------------------
        # Embedding Model
        # ---------------------------------------

        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        # ---------------------------------------
        # Groq Client
        # ---------------------------------------

        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )

        # ---------------------------------------
        # FAISS
        # ---------------------------------------

        faiss_index = faiss.read_index("rag/vectorstore/faiss.index")

        vector_store = FaissVectorStore(
            faiss_index=faiss_index
        )

        storage_context = StorageContext.from_defaults(
            persist_dir="rag/storage",
            vector_store=vector_store
        )

        index = load_index_from_storage(
            storage_context
        )

        # ---------------------------------------
        # Retriever
        # ---------------------------------------

        self.retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5,
            vector_store_query_mode=VectorStoreQueryMode.MMR,
            mmr_threshold=0.5
        )

    # -------------------------------------------------
    # Call LLM
    # -------------------------------------------------

    def call_llm(
        self,
        prompt,
        temperature=0,
        max_tokens=500
    ):

        response = self.client.chat.completions.create(

            model=self.model,

            temperature=temperature,

            max_tokens=max_tokens,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content.strip()

    # -------------------------------------------------
    # Rewrite Query
    # -------------------------------------------------

    def rewrite_query(
        self,
        query,
        chat_history
    ):

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

        revised_query = self.call_llm(
            rewrite_prompt,
            temperature=0,
            max_tokens=100
        )

        print("\nStandalone Query:")
        print(revised_query)

        return revised_query

    # -------------------------------------------------
    # Generate Multiple Queries
    # -------------------------------------------------

    def generate_multi_queries(
        self,
        revised_query
    ):

        multi_query_prompt = f"""
    Generate 3 different search queries for the following question.

    The alternative queries should retrieve complementary information while preserving
    the original meaning.

    Original Query:
    {revised_query}

    Return ONLY the queries, one per line.
    """

        multi_query_response = self.call_llm(
            multi_query_prompt,
            temperature=0,
            max_tokens=150
        )

        query_variations = [
            revised_query
        ]

        for q in multi_query_response.split("\n"):

            if q.strip():

                cleaned_query = (
                    q.strip("-• ").strip()
                )

                query_variations.append(
                    cleaned_query
                )

        print("\nGenerated Queries")
        print("-" * 70)

        for i, q in enumerate(query_variations, 1):
            print(f"{i}. {q}")

        return query_variations

    # -------------------------------------------------
    # Retrieve Chunks
    # -------------------------------------------------

    def retrieve_chunks(
        self,
        query_variations
    ):

        all_contexts = []

        for q in query_variations:

            # -------------------------
            # FAISS Retrieval
            # -------------------------

            nodes = self.retriever.retrieve(q)

            # -------------------------
            # BM25 Retrieval
            # -------------------------

            tokenized_query = q.lower().split()

            scores = self.bm25.get_scores(
                tokenized_query
            )

            # Find Top 5 BM25 Results

            top_indices = []

            temp_scores = scores.copy()

            for _ in range(5):

                highest_score = -1
                highest_index = -1

                for i in range(len(temp_scores)):

                    if temp_scores[i] > highest_score:

                        highest_score = temp_scores[i]
                        highest_index = i

                top_indices.append(
                    highest_index
                )

                temp_scores[highest_index] = -1

            bm25_nodes = []

            for idx in top_indices:

                bm25_nodes.append(
                    self.all_nodes[idx]
                )

            query_context = []

            # FAISS

            for node in nodes:

                query_context.append(
                    node.text
                )

            # BM25

            for node in bm25_nodes:

                query_context.append(
                    node.text
                )

            all_contexts.append(
                query_context
            )

        return all_contexts
    # -------------------------------------------------
    # Rerank Chunks
    # -------------------------------------------------

    def rerank_chunks(
        self,
        revised_query,
        all_contexts
    ):

        rrf_scores = {}

        unique_chunks = []

        k = 60

        # ---------------------------------------
        # Reciprocal Rank Fusion
        # ---------------------------------------

        for query_context in all_contexts:

            rank = 1

            for chunk in query_context:

                if chunk not in unique_chunks:

                    unique_chunks.append(
                        chunk
                    )

                if chunk not in rrf_scores:

                    rrf_scores[chunk] = 0

                rrf_scores[chunk] += (
                    1 / (k + rank)
                )

                rank += 1

        fused_chunks = unique_chunks.copy()

        for i in range(len(fused_chunks)):

            best = i

            for j in range(i + 1, len(fused_chunks)):

                if (
                    rrf_scores[fused_chunks[j]]
                    >
                    rrf_scores[fused_chunks[best]]
                ):

                    best = j

            fused_chunks[i], fused_chunks[best] = (
                fused_chunks[best],
                fused_chunks[i]
            )

        print("\nChunks After Reciprocal Rank Fusion")
        print("-" * 70)

        for i, chunk in enumerate(fused_chunks, 1):

            print(f"\nChunk {i}")

            print(
                "Score:",
                round(rrf_scores[chunk], 5)
            )

            print(chunk[:250] + "...")

        # ---------------------------------------
        # Cohere Reranking
        # ---------------------------------------

        rerank_response = self.co.rerank(

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

            reranked_chunks.append(
                chunk
            )

            print(
                f"\nRelevance Score: "
                f"{result.relevance_score:.4f}"
            )

            print(chunk[:250] + "...")

        return reranked_chunks
    # -------------------------------------------------
    # Build Prompt
    # -------------------------------------------------

    def build_prompt(
        self,
        query,
        reranked_chunks
    ):

        prompt = f"""
You are an intelligent AI assistant.

Answer the user's question ONLY using the retrieved context below.

If the retrieved context does not contain enough information to answer the
question, simply reply:

"I could not find the answer in the provided documents."

Retrieved Context:

{reranked_chunks}

User Question:

{query}
"""

        return prompt
    # -------------------------------------------------
    # Generate Final Answer
    # -------------------------------------------------

    def generate_answer(
        self,
        prompt
    ):

        answer = self.call_llm(
            prompt=prompt,
            temperature=0,
            max_tokens=500
        )

        return answer

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def answer(
        self,
        question,
        chat_history
    ):

        revised_query = self.rewrite_query(
            query=question,
            chat_history=chat_history
        )

        query_variations = self.generate_multi_queries(
            revised_query
        )

        all_contexts = self.retrieve_chunks(
            query_variations
        )

        reranked_chunks = self.rerank_chunks(
            revised_query,
            all_contexts
        )

        prompt = self.build_prompt(
            query=question,
            reranked_chunks=reranked_chunks
        )

        answer = self.generate_answer(
            prompt
        )

        return answer