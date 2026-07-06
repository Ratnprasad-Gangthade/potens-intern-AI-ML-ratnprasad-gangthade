import json
import os

from langchain_community.vectorstores import FAISS

from rag_flow.language_handling import llm


COMPARISON_VERDICTS = {
    "Agree",
    "Partially Agree",
    "Contradict",
    "Have Insufficient Information",
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
