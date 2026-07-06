import os

from langchain_community.vectorstores import FAISS

from rag_flow.language_handling import (
    detect_language,
    llm,
    translate_from_english,
    translate_to_english,
)


INSUFFICIENT_INFORMATION_RESPONSE = (
    "The uploaded research papers do not contain enough "
    "information to answer this question."
)


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
