import faiss
import pickle

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

    Responsibilities:
    - Load documents
    - Chunk documents
    - Build FAISS index
    - Save index and metadata
    """

    def __init__(self):

        load_dotenv()

        self.dimension = 384
        self.documents = None
        self.nodes = None

        self.configure_embedding_model()

    # -------------------------------------------------
    # Configure Embedding Model
    # -------------------------------------------------

    def configure_embedding_model(self):

        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    # -------------------------------------------------
    # Load Documents
    # -------------------------------------------------

    def load_documents(self, input_dir):

        print("\nLoading documents...\n")

        self.documents = SimpleDirectoryReader(
            input_dir=input_dir,
            required_exts=[".txt"]
        ).load_data()

        print(f"Loaded {len(self.documents)} documents.\n")

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

        with open("nodes.pkl", "wb") as f:
            pickle.dump(self.nodes, f)

        print(f"\nCreated {len(self.nodes)} chunks.\n")

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

        storage_context.persist("storage")

        faiss.write_index(
            faiss_index,
            "faiss.index"
        )

        print("\nFAISS index successfully created.")

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def ingest(self, input_dir="docs"):

        self.load_documents(input_dir)

        self.chunk_documents()

        self.build_index()