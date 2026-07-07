from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_flow.document_loader import load_documents


FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "gemini-embedding-2-preview"


def build_vector_db(documents_folder: str) -> FAISS:
    """
    Build a FAISS vector database from all PDFs present
    inside the documents folder.
    """

    docs = load_documents(documents_folder)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vector_db.save_local(FAISS_INDEX_PATH)

    return vector_db


def load_vector_db(index_path: str = FAISS_INDEX_PATH) -> FAISS:
    """
    Load an existing FAISS vector database from disk.
    """

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
