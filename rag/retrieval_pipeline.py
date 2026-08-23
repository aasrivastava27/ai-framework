import os
import faiss
import cohere
import numpy as np

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
)

from llama_index.core.node_parser import SentenceSplitter

from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding
)

from rank_bm25 import BM25Okapi


class RAG:
    """
    User-specific Retrieval-Augmented Generation pipeline.

    Each RAG instance belongs to one Google user.

    Documents are loaded from:

        rag/docs/<user_google_email>/

    Example:

        rag/docs/aashutoshsr27@gmail.com/
    """

    # =================================================
    # Initialize
    # =================================================

    def __init__(self, user_google_email: str):

        if not user_google_email:

            raise ValueError(
                "Google email is required for RAG."
            )

        self.user_google_email = (
            user_google_email.strip().lower()
        )

        # ---------------------------------------------
        # User's document directory
        # ---------------------------------------------

        self.docs_dir = (
            Path("rag")
            / "docs"
            / self.user_google_email
        )

        self.docs_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ---------------------------------------------
        # RAG components
        # ---------------------------------------------

        self.client = None
        self.co = None

        # Direct FAISS index
        self.faiss_index = None

        # Embedding model
        self.embed_model = None

        # BM25
        self.bm25 = None

        # Document chunks
        self.all_nodes = []

        self.available_files = []

        # ---------------------------------------------
        # Groq model
        # ---------------------------------------------

        self.model = "openai/gpt-oss-120b"

        # ---------------------------------------------
        # Used to detect when documents change
        # ---------------------------------------------

        self.document_signature = None

        # ---------------------------------------------
        # Load models
        # ---------------------------------------------

        self.load_models()

        # ---------------------------------------------
        # Build initial user index
        # ---------------------------------------------

        self.refresh_index()


    # =================================================
    # Load Models
    # =================================================

    def load_models(self):

        load_dotenv()

        # ---------------------------------------------
        # Embedding model
        # ---------------------------------------------

        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        # Also register it with LlamaIndex
        Settings.embed_model = self.embed_model

        # ---------------------------------------------
        # Cohere
        # ---------------------------------------------

        self.co = cohere.Client(
            os.getenv("COHERE_API_KEY")
        )

        # ---------------------------------------------
        # Groq / OpenAI-compatible client
        # ---------------------------------------------

        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )


    # =================================================
    # Get User Documents
    # =================================================

    def get_document_files(self):

        return sorted(
            self.docs_dir.glob("*.txt")
        )


    # =================================================
    # Get Document Signature
    # =================================================

    def get_document_signature(self):

        files = self.get_document_files()

        return tuple(
            (
                file.name,
                file.stat().st_mtime_ns,
                file.stat().st_size,
            )
            for file in files
        )


    # =================================================
    # Refresh User Index
    # =================================================

    def refresh_index(self):

        files = self.get_document_files()

        self.available_files = [
            file.name
            for file in files
        ]

        current_signature = (
            self.get_document_signature()
        )

        # ---------------------------------------------
        # Nothing changed
        # ---------------------------------------------

        if (
            self.document_signature is not None
            and
            current_signature == self.document_signature
        ):

            return


        # ---------------------------------------------
        # No documents
        # ---------------------------------------------

        if not files:

            print(
                f"[RAG] No documents found for "
                f"{self.user_google_email}"
            )

            self.all_nodes = []
            self.bm25 = None
            self.faiss_index = None

            self.document_signature = (
                current_signature
            )

            return


        print(
            f"[RAG] Loading documents for "
            f"{self.user_google_email}"
        )


        # ---------------------------------------------
        # Load documents
        # ---------------------------------------------

        documents = SimpleDirectoryReader(
            input_dir=str(self.docs_dir),
            required_exts=[".txt"]
        ).load_data()


        print(
            f"[RAG] Loaded {len(documents)} documents"
        )


        # ---------------------------------------------
        # Split documents into chunks
        # ---------------------------------------------

        splitter = SentenceSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        self.all_nodes = (
            splitter.get_nodes_from_documents(
                documents
            )
        )


        print(
            f"[RAG] Created "
            f"{len(self.all_nodes)} chunks"
        )


        # =================================================
        # BM25
        # =================================================

        tokenized_docs = []

        for node in self.all_nodes:

            tokenized_docs.append(
                node.text.lower().split()
            )


        self.bm25 = BM25Okapi(
            tokenized_docs
        )


        # =================================================
        # FAISS
        # =================================================

        # BAAI/bge-small-en-v1.5 produces
        # 384-dimensional embeddings.

        dimension = 384

        self.faiss_index = faiss.IndexFlatL2(
            dimension
        )


        print(
            "[RAG] Creating FAISS embeddings..."
        )


        # ---------------------------------------------
        # Create embeddings for every chunk
        # ---------------------------------------------

        embeddings = []

        for node in self.all_nodes:

            embedding = (
                self.embed_model.get_text_embedding(
                    node.text
                )
            )

            embeddings.append(
                embedding
            )


        # ---------------------------------------------
        # Convert embeddings to NumPy matrix
        # ---------------------------------------------

        embedding_matrix = np.asarray(
            embeddings,
            dtype="float32"
        )


        # ---------------------------------------------
        # Safety check
        # ---------------------------------------------

        if (
            embedding_matrix.ndim != 2
            or
            embedding_matrix.shape[1] != dimension
        ):

            raise ValueError(
                "Invalid embedding matrix shape: "
                f"{embedding_matrix.shape}. "
                f"Expected (?, {dimension})."
            )


        # ---------------------------------------------
        # Add embeddings to FAISS
        # ---------------------------------------------

        self.faiss_index.add(
            embedding_matrix
        )


        print(
            f"[RAG] FAISS index contains "
            f"{self.faiss_index.ntotal} vectors"
        )


        # ---------------------------------------------
        # Remember document state
        # ---------------------------------------------

        self.document_signature = (
            current_signature
        )


        print(
            f"[RAG] Index ready for "
            f"{self.user_google_email}"
        )


    # =================================================
    # Call LLM
    # =================================================

    def call_llm(
        self,
        prompt,
        temperature=0,
        max_tokens=500
    ):

        response = (
            self.client.chat.completions.create(
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
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )


    # =================================================
    # Rewrite Query
    # =================================================

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

Rewrite the latest question into a standalone
question by resolving references such as
"it", "they", "that company", etc.

If the latest question is already standalone,
return it unchanged.

Return ONLY the rewritten question.
"""

        revised_query = self.call_llm(
            rewrite_prompt,
            temperature=0,
            max_tokens=100
        )

        # ---------------------------------------------
        # Safety fallback
        # ---------------------------------------------

        if not revised_query.strip():

            revised_query = query


        print(
            "\nStandalone Query:"
        )

        print(revised_query)

        return revised_query


    # =================================================
    # Generate Multiple Queries
    # =================================================

    def generate_multi_queries(
        self,
        revised_query
    ):

        multi_query_prompt = f"""
Generate 3 different search queries for the
following question.

The alternative queries should retrieve
complementary information while preserving
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


        for q in (
            multi_query_response.split("\n")
        ):

            if q.strip():

                cleaned_query = (
                    q.strip("-• ").strip()
                )

                if cleaned_query:

                    query_variations.append(
                        cleaned_query
                    )


        print(
            "\nGenerated Queries"
        )

        print("-" * 70)


        for i, q in enumerate(
            query_variations,
            1
        ):

            print(
                f"{i}. {q}"
            )


        return query_variations


    # =================================================
    # Retrieve Chunks
    # =================================================

    def retrieve_chunks(
        self,
        query_variations
    ):

        all_contexts = []


        for q in query_variations:

            query_context = []


            # =================================================
            # FAISS Retrieval
            # =================================================

            if (
                self.faiss_index is not None
                and
                self.faiss_index.ntotal > 0
            ):

                # ---------------------------------------------
                # Create query embedding
                # ---------------------------------------------

                query_embedding = (
                    self.embed_model.get_query_embedding(
                        q
                    )
                )


                query_embedding = np.asarray(
                    query_embedding,
                    dtype="float32"
                )


                # ---------------------------------------------
                # Safety check
                # ---------------------------------------------

                if query_embedding.ndim != 1:

                    query_embedding = (
                        query_embedding.reshape(-1)
                    )


                if len(query_embedding) != 384:

                    raise ValueError(
                        "Invalid query embedding size: "
                        f"{len(query_embedding)}. "
                        "Expected 384."
                    )


                # ---------------------------------------------
                # FAISS expects shape:
                #
                # (number_of_queries, dimension)
                #
                # So convert:
                #
                # (384,)
                #
                # into:
                #
                # (1, 384)
                # ---------------------------------------------

                query_vector = (
                    query_embedding.reshape(
                        1,
                        -1
                    )
                )


                # ---------------------------------------------
                # Search FAISS
                # ---------------------------------------------

                distances, indices = (
                    self.faiss_index.search(
                        query_vector,
                        min(
                            5,
                            self.faiss_index.ntotal
                        )
                    )
                )


                # ---------------------------------------------
                # Add retrieved chunks
                # ---------------------------------------------

                for idx in indices[0]:

                    if idx == -1:
                        continue

                    query_context.append(
                        self.all_nodes[
                            int(idx)
                        ].text
                    )


            # =================================================
            # BM25 Retrieval
            # =================================================

            if self.bm25 is not None:

                tokenized_query = (
                    q.lower().split()
                )

                scores = (
                    self.bm25.get_scores(
                        tokenized_query
                    )
                )


                top_indices = []

                temp_scores = scores.copy()


                number_to_retrieve = min(
                    5,
                    len(temp_scores)
                )


                for _ in range(
                    number_to_retrieve
                ):

                    highest_score = -1
                    highest_index = -1


                    for i in range(
                        len(temp_scores)
                    ):

                        if (
                            temp_scores[i]
                            > highest_score
                        ):

                            highest_score = (
                                temp_scores[i]
                            )

                            highest_index = i


                    if highest_index == -1:
                        break


                    top_indices.append(
                        highest_index
                    )

                    temp_scores[
                        highest_index
                    ] = -1


                for idx in top_indices:

                    query_context.append(
                        self.all_nodes[
                            idx
                        ].text
                    )


            # ---------------------------------------------
            # Save results for this query
            # ---------------------------------------------

            all_contexts.append(
                query_context
            )


        return all_contexts


    # =================================================
    # Rerank Chunks
    # =================================================

    def rerank_chunks(
        self,
        revised_query,
        all_contexts
    ):

        rrf_scores = {}

        unique_chunks = []

        k = 60


        # =================================================
        # Reciprocal Rank Fusion
        # =================================================

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


        # =================================================
        # Sort chunks
        # =================================================

        fused_chunks = (
            unique_chunks.copy()
        )


        for i in range(
            len(fused_chunks)
        ):

            best = i


            for j in range(
                i + 1,
                len(fused_chunks)
            ):

                if (
                    rrf_scores[
                        fused_chunks[j]
                    ]
                    >
                    rrf_scores[
                        fused_chunks[best]
                    ]
                ):

                    best = j


            fused_chunks[i], fused_chunks[best] = (
                fused_chunks[best],
                fused_chunks[i]
            )


        if not fused_chunks:

            return []


        # =================================================
        # Cohere Reranking
        # =================================================

        if not revised_query.strip():

            return fused_chunks[:5]


        rerank_response = self.co.rerank(
            model="rerank-v3.5",
            query=revised_query,
            documents=fused_chunks,
            top_n=min(
                5,
                len(fused_chunks)
            )
        )


        reranked_chunks = []


        for result in (
            rerank_response.results
        ):

            chunk = (
                fused_chunks[result.index]
            )

            reranked_chunks.append(
                chunk
            )


        return reranked_chunks


    # =================================================
    # Build Prompt
    # =================================================

    def build_prompt(
        self,
        query,
        reranked_chunks
    ):

        prompt = f"""
You are an intelligent AI assistant.

Primarily use the retrieved contexts below
to answer the user's question.

The contexts come only from the user's own
uploaded documents.

If the retrieved contexts are not enough to
answer the question, say:

"I could not find the answer in your
uploaded documents."

Do not invent information.

Retrieved Contexts:
{reranked_chunks}

Original User Question:
{query}
"""

        return prompt


    # =================================================
    # Answer
    # =================================================

    def answer(
        self,
        question: str,
        chat_history=""
    ) -> str:

        # ---------------------------------------------
        # Check whether new documents were uploaded
        # ---------------------------------------------

        self.refresh_index()


        # ---------------------------------------------
        # No documents
        # ---------------------------------------------

        if not self.all_nodes:

            return (
                "You don't have any uploaded "
                "documents yet. Please upload a "
                ".txt document first."
            )


        # ---------------------------------------------
        # Rewrite query
        # ---------------------------------------------

        revised_query = self.rewrite_query(
            query=question,
            chat_history=chat_history
        )


        # ---------------------------------------------
        # Generate multiple queries
        # ---------------------------------------------

        query_variations = (
            self.generate_multi_queries(
                revised_query
            )
        )


        # ---------------------------------------------
        # Retrieve
        # ---------------------------------------------

        all_contexts = (
            self.retrieve_chunks(
                query_variations
            )
        )


        # ---------------------------------------------
        # Rerank
        # ---------------------------------------------

        reranked_chunks = (
            self.rerank_chunks(
                revised_query,
                all_contexts
            )
        )


        # ---------------------------------------------
        # No relevant chunks
        # ---------------------------------------------

        if not reranked_chunks:

            return (
                "I could not find relevant "
                "information in your uploaded "
                "documents."
            )


        # ---------------------------------------------
        # Build final prompt
        # ---------------------------------------------

        prompt = self.build_prompt(
            query=question,
            reranked_chunks=reranked_chunks
        )


        # ---------------------------------------------
        # Generate final answer
        # ---------------------------------------------

        answer = self.call_llm(
            prompt
        )


        return answer