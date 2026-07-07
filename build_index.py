from rag_flow.vector_store import build_vector_db

print("Building FAISS Index...")

build_vector_db("Documents")

print("FAISS Index Created Successfully!")