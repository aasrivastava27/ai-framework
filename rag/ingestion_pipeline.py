import os
import pickle
from pathlib import Path

import faiss

from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


class IngestionPipeline:
    """
    Handles the complete document ingestion process.

    Each user's documents and vector store are kept separate.

    Directory structure:

        rag/
        ├── docs/
        │   ├── user1@gmail.com/
        │   │   ├── document1.txt
        │   │   └── document2.txt
        │   │
        │   └── user2@gmail.com/
        │       └── document1.txt
        │
        ├── vectorstore/
        │   ├── user1@gmail.com/
        │   │   ├── nodes.pkl
        │   │   └── faiss.index
        │   │
        │   └── user2@gmail.com/
        │       ├── nodes.pkl
        │       └── faiss.index
        │
        └── storage/
            ├── user1@gmail.com/
            └── user2@gmail.com/

    Responsibilities:
    - Load documents for one user
    - Chunk documents
    - Build FAISS index
    - Save user-specific index and metadata
    """

    def __init__(self):

        load_dotenv()

        self.dimension = 384

        self.documents = None
        self.nodes = None

        self.user_google_email = None

        self.docs_dir = None
        self.vectorstore_dir = None
        self.storage_dir = None

        self.configure_embedding_model()

    # -------------------------------------------------
    # Configure Embedding Model
    # -------------------------------------------------

    def configure_embedding_model(self):

        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    # -------------------------------------------------
    # Configure User Directories
    # -------------------------------------------------

    def configure_user_directories(
        self,
        user_google_email: str
    ):

        self.user_google_email = user_google_email.strip().lower()

        project_root = Path(__file__).resolve().parent.parent

        # -------------------------------------------------
        # User documents
        # -------------------------------------------------

        self.docs_dir = (
            project_root
            / "rag"
            / "docs"
            / self.user_google_email
        )

        # -------------------------------------------------
        # User vector store
        # -------------------------------------------------

        self.vectorstore_dir = (
            project_root
            / "rag"
            / "vectorstore"
            / self.user_google_email
        )

        # -------------------------------------------------
        # User LlamaIndex storage
        # -------------------------------------------------

        self.storage_dir = (
            project_root
            / "rag"
            / "storage"
            / self.user_google_email
        )

        # Create directories if they don't exist.

        self.docs_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.vectorstore_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # -------------------------------------------------
    # Load Documents
    # -------------------------------------------------

    def load_documents(self):

        print(
            f"\nLoading documents for "
            f"{self.user_google_email}...\n"
        )

        self.documents = SimpleDirectoryReader(
            input_dir=str(self.docs_dir),
            required_exts=[".txt"]
        ).load_data()

        print(
            f"Loaded {len(self.documents)} documents.\n"
        )

        for i, doc in enumerate(self.documents):

            print(f"Document {i + 1}")

            print(doc.metadata)

            print(doc.text[:100])

            print("-" * 60)

    # -------------------------------------------------
    # Chunk Documents
    # -------------------------------------------------

    def chunk_documents(self):

        splitter = SentenceSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        self.nodes = splitter.get_nodes_from_documents(
            self.documents
        )

        nodes_path = (
            self.vectorstore_dir
            / "nodes.pkl"
        )

        with open(
            nodes_path,
            "wb"
        ) as f:

            pickle.dump(
                self.nodes,
                f
            )

        print(
            f"\nCreated {len(self.nodes)} chunks."
        )

    # -------------------------------------------------
    # Build FAISS Index
    # -------------------------------------------------

    def build_index(self):

        faiss_index = faiss.IndexFlatL2(
            self.dimension
        )

        vector_store = FaissVectorStore(
            faiss_index=faiss_index
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        VectorStoreIndex(
            self.nodes,
            storage_context=storage_context
        )

        # -------------------------------------------------
        # Persist LlamaIndex storage
        # -------------------------------------------------

        storage_context.persist(
            persist_dir=str(
                self.storage_dir
            )
        )

        # -------------------------------------------------
        # Save FAISS index
        # -------------------------------------------------

        faiss_path = (
            self.vectorstore_dir
            / "faiss.index"
        )

        faiss.write_index(
            faiss_index,
            str(faiss_path)
        )

        print(
            f"\nFAISS index successfully created "
            f"for {self.user_google_email}."
        )

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def ingest(
        self,
        user_google_email: str
    ):

        # Configure all paths for this user.

        self.configure_user_directories(
            user_google_email
        )

        # Load only this user's documents.

        self.load_documents()

        # Chunk only this user's documents.

        self.chunk_documents()

        # Build only this user's FAISS index.

        self.build_index()
