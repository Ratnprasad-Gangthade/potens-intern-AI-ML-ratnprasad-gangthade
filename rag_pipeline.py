import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

INSUFFICIENT_INFORMATION_RESPONSE = (
    "The uploaded research papers do not contain enough "
    "information to answer this question."
)


def load_documents(documents_folder: str):
    """
    Load all PDF documents from the given folder.
    """

    all_docs = []

    for file_name in os.listdir(documents_folder):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(documents_folder, file_name)

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            all_docs.extend(docs)

    return all_docs


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

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vector_db.save_local("faiss_index")

    return vector_db


def answer_query(vector_db: FAISS, query: str, k: int = 2) -> dict:
    """
    Retrieve relevant chunks and generate an answer with citations.
    """

    documents = vector_db.similarity_search(
        query=query,
        k=k
    )

    context = ""

    for doc in documents:
        context += doc.page_content + "\n"

    sufficiency_prompt = f"""
You are a strict context sufficiency checker.

Decide whether the provided context contains enough information
to answer the question directly.

Rules:
- Use ONLY the provided context.
- Do not use outside knowledge.
- Do not infer facts that are not supported by the context.
- If the context is unrelated, incomplete, or insufficient, answer NO.
- Answer with exactly one word: YES or NO.

Context:
{context}

Question:
{query}
"""

    sufficiency_result = llm.invoke(sufficiency_prompt)
    is_context_sufficient = sufficiency_result.content.strip().upper() == "YES"

    citations = []

    for doc in documents:
        citations.append({
            "source_file": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "snippet": doc.page_content[:200],
        })

    if not is_context_sufficient:
        return {
            "answer": INSUFFICIENT_INFORMATION_RESPONSE,
            "citations": citations,
        }

    prompt = f"""
You are a Research Paper Analysis Assistant.

Answer ONLY using the provided context.

Rules:
- Do not use outside knowledge.
- Do not guess.
- If the answer is not present in the retrieved context,
  explicitly say:
  "The uploaded research papers do not contain enough
  information to answer this question."

Context:
{context}

Question:
{query}
"""

    result = llm.invoke(prompt)

    return {
        "answer": result.content,
        "citations": citations,
    }
