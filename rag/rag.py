import os
import pickle
import faiss

from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.vector_stores.faiss import FaissVectorStore

from llama_index.core.retrievers import VectorIndexRetriever

from llama_index.core.vector_stores.types import (
    VectorStoreQueryMode
)

from llama_index.core import SimpleDirectoryReader


class RAG:

    def __init__(self):

        # -------------------------------------------------
        # Base directories
        # -------------------------------------------------

        self.docs_root = Path("rag") / "docs"

        self.data_root = Path("rag") / "data"


        # -------------------------------------------------
        # Embedding model
        # -------------------------------------------------

        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )


    # =====================================================
    # Build / Load User Index
    # =====================================================

    def load_user_index(self, google_email: str):

        # -------------------------------------------------
        # User-specific directories
        # -------------------------------------------------

        user_docs_dir = (
            self.docs_root
            / google_email
        )

        user_data_dir = (
            self.data_root
            / google_email
        )


        # -------------------------------------------------
        # Make sure user's document directory exists
        # -------------------------------------------------

        user_docs_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        # -------------------------------------------------
        # Check whether user has documents
        # -------------------------------------------------

        txt_files = list(
            user_docs_dir.glob("*.txt")
        )

        if not txt_files:

            return None


        # -------------------------------------------------
        # Load user's documents
        # -------------------------------------------------

        documents = SimpleDirectoryReader(
            input_dir=str(user_docs_dir),
            required_exts=[".txt"]
        ).load_data()


        # -------------------------------------------------
        # Create user data directory
        # -------------------------------------------------

        user_data_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        # -------------------------------------------------
        # Create FAISS index
        # -------------------------------------------------

        dimension = 384

        faiss_index = faiss.IndexFlatL2(
            dimension
        )


        vector_store = FaissVectorStore(
            faiss_index=faiss_index
        )


        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )


        # -------------------------------------------------
        # Create LlamaIndex
        # -------------------------------------------------

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )


        # -------------------------------------------------
        # Persist user-specific index
        # -------------------------------------------------

        storage_context.persist(
            persist_dir=str(user_data_dir)
        )


        faiss.write_index(
            faiss_index,
            str(user_data_dir / "faiss.index")
        )


        # -------------------------------------------------
        # Create retriever
        # -------------------------------------------------

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5,
            vector_store_query_mode=VectorStoreQueryMode.MMR,
            mmr_threshold=0.5
        )


        return retriever


    # =====================================================
    # Answer Question
    # =====================================================

    def answer(
        self,
        question: str,
        google_email: str
    ) -> str:

        # -------------------------------------------------
        # Load user's RAG index
        # -------------------------------------------------

        retriever = self.load_user_index(
            google_email
        )


        # -------------------------------------------------
        # No documents
        # -------------------------------------------------

        if retriever is None:

            return (
                "You don't have any uploaded documents yet. "
                "Please upload a .txt document first."
            )


        # -------------------------------------------------
        # Retrieve relevant chunks
        # -------------------------------------------------

        nodes = retriever.retrieve(
            question
        )


        # -------------------------------------------------
        # No relevant information
        # -------------------------------------------------

        if not nodes:

            return (
                "I could not find relevant information "
                "in your uploaded documents."
            )


        # -------------------------------------------------
        # Build context
        # -------------------------------------------------

        context = "\n\n".join(
            node.node.text
            for node in nodes
        )


        # -------------------------------------------------
        # Return retrieved information for now
        # -------------------------------------------------

        return context