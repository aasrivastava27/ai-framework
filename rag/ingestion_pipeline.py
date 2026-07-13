import os
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

load_dotenv()

# ---------------------------------------
# Configure embedding model
# ---------------------------------------

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# ---------------------------------------
# Load documents
# ---------------------------------------

print("Loading documents...\n")

documents = SimpleDirectoryReader(
    input_dir="docs",
    required_exts=[".txt"]
).load_data()

print(f"Loaded {len(documents)} documents.\n")

for i, doc in enumerate(documents):
    print(f"Document {i+1}")
    print(doc.metadata)
    print(doc.text[:100])
    print("-" * 60)

# ---------------------------------------
# Chunk documents
# ---------------------------------------

splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

nodes = splitter.get_nodes_from_documents(documents)

with open("nodes.pkl", "wb") as f:
    pickle.dump(nodes, f)

print(f"\nCreated {len(nodes)} chunks.\n")

# ---------------------------------------
# Create FAISS index
# ---------------------------------------

dimension = 384

faiss_index = faiss.IndexFlatL2(dimension)

vector_store = FaissVectorStore(
    faiss_index=faiss_index
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context
)

# ---------------------------------------
# Persist storage
# ---------------------------------------

storage_context.persist("storage")

faiss.write_index(
    faiss_index,
    "faiss.index"
)

print("\nFAISS index successfully created.")