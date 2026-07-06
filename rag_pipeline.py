import os
import json
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

COMPARISON_VERDICTS = {
    "Agree",
    "Partially Agree",
    "Contradict",
    "Have Insufficient Information",
}


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.
    """

    prompt = f"""
Detect the language of the following text.

Rules:
- Return only the English name of the language.
- Do not explain.

Text:
{text}
"""

    result = llm.invoke(prompt)

    return result.content.strip()


def translate_to_english(text: str) -> str:
    """
    Translate text to English.
    """

    prompt = f"""
Translate the following text to English.

Rules:
- Return only the translated text.
- Do not answer the text.
- Do not explain.

Text:
{text}
"""

    result = llm.invoke(prompt)

    return result.content.strip()


def translate_from_english(text: str, target_language: str) -> str:
    """
    Translate English text to the target language.
    """

    prompt = f"""
Translate the following English text to {target_language}.

Rules:
- Return only the translated text.
- Do not add explanations.

Text:
{text}
"""

    result = llm.invoke(prompt)

    return result.content.strip()


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

    query_language = detect_language(query).strip().strip(".")
    is_english_query = query_language.lower() == "english"
    english_query = query if is_english_query else translate_to_english(query)

    documents = vector_db.similarity_search(
        query=english_query,
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
{english_query}
"""

    sufficiency_result = llm.invoke(sufficiency_prompt)
    is_context_sufficient = sufficiency_result.content.strip().upper() == "YES"

    citations = []

    for doc in documents:
        citations.append({
            "source_file": os.path.basename(
                doc.metadata.get("source", "")
            ),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "snippet": doc.page_content[:200],
        })

    if not is_context_sufficient:
        answer = INSUFFICIENT_INFORMATION_RESPONSE

        if not is_english_query:
            answer = translate_from_english(answer, query_language)

        return {
            "answer": answer,
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
{english_query}
"""

    result = llm.invoke(prompt)
    answer = result.content

    if not is_english_query:
        answer = translate_from_english(answer, query_language)

    return {
        "answer": answer,
        "citations": citations,
    }


def compare_documents(
    vector_db: FAISS,
    document_a: str,
    document_b: str,
    topic: str,
    k: int = 4
) -> dict:
    """
    Compare two documents on a topic using only retrieved chunks.
    """

    documents_a = vector_db.similarity_search(
        query=topic,
        k=k,
        filter={"source": document_a}
    )

    documents_b = vector_db.similarity_search(
        query=topic,
        k=k,
        filter={"source": document_b}
    )

    context_a = ""
    context_b = ""

    for doc in documents_a:
        context_a += doc.page_content + "\n"

    for doc in documents_b:
        context_b += doc.page_content + "\n"

    evidence_from_document_a = []
    evidence_from_document_b = []

    for doc in documents_a:
        evidence_from_document_a.append({
            "source_file": os.path.basename(
                doc.metadata.get("source", "")
            ),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "snippet": doc.page_content[:200],
        })

    for doc in documents_b:
        evidence_from_document_b.append({
            "source_file": os.path.basename(
                doc.metadata.get("source", "")
            ),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "snippet": doc.page_content[:200],
        })

    if not documents_a or not documents_b:
        return {
            "verdict": "Have Insufficient Information",
            "explanation": (
                "One or both documents do not contain retrieved context "
                "relevant to the requested topic."
            ),
            "evidence_from_document_a": evidence_from_document_a,
            "evidence_from_document_b": evidence_from_document_b,
        }

    prompt = f"""
You are a Research Paper Comparison Assistant.

Compare Document A and Document B only on the requested topic.

Rules:
- Use ONLY the provided context.
- Do not use outside knowledge.
- Do not compare the entire PDFs.
- Do not guess or infer facts that are not supported by the context.
- If one or both documents do not contain enough relevant information,
  use the verdict: Have Insufficient Information.
- The verdict must be exactly one of:
  Agree
  Partially Agree
  Contradict
  Have Insufficient Information
- Return only valid JSON with these keys:
  verdict
  explanation

Document A Context:
{context_a}

Document B Context:
{context_b}

Topic:
{topic}
"""

    result = llm.invoke(prompt)
    comparison_text = result.content.strip()

    if comparison_text.startswith("```json"):
        comparison_text = comparison_text[7:]
    elif comparison_text.startswith("```"):
        comparison_text = comparison_text[3:]

    if comparison_text.endswith("```"):
        comparison_text = comparison_text[:-3]

    try:
        comparison = json.loads(comparison_text.strip())
    except json.JSONDecodeError:
        return {
            "verdict": "Have Insufficient Information",
            "explanation": "The model returned an invalid comparison format.",
            "evidence_from_document_a": evidence_from_document_a,
            "evidence_from_document_b": evidence_from_document_b,
        }

    verdict = comparison.get("verdict")

    if verdict not in COMPARISON_VERDICTS:
        verdict = "Have Insufficient Information"

    return {
        "verdict": verdict,
        "explanation": comparison.get("explanation", ""),
        "evidence_from_document_a": evidence_from_document_a,
        "evidence_from_document_b": evidence_from_document_b,
    }
