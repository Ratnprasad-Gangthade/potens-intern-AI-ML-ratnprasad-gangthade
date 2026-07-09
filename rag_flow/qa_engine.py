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


def answer_query(vector_db: FAISS, query: str, k: int = 3) -> dict:
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
You are a context sufficiency checker.

Determine whether the provided context contains enough information
to answer the user's question, even if the answer is only partial.

Rules:
- Use ONLY the provided context.
- Do not use outside knowledge.
- If the context contains relevant information that can reasonably answer the question, answer YES.
- Answer NO only if the context contains no relevant information or is clearly insufficient to answer the question at all.
- Respond with exactly one word:
YES
or
NO.

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
Use only the provided context.

If the context partially answers the question,
provide the partial answer and explicitly mention what information is missing.

Only return:

"The uploaded research papers do not contain enough information..."

if none of the retrieved context is relevant.

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
