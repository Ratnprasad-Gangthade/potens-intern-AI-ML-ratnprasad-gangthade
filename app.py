import os
import streamlit as st

from rag_flow.vector_store import load_vector_db
from rag_flow.qa_engine import answer_query
from rag_flow.contradiction import compare_documents

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Potens RAG Pipeline",
    page_icon="📄",
    layout="wide"
)

DOCUMENTS_FOLDER = "Documents"

# -------------------------------------------------------
# Load Vector Database
# -------------------------------------------------------

@st.cache_resource
def get_vector_db():
    return load_vector_db()


try:
    vector_db = get_vector_db()
except Exception:
    st.error(
        "Unable to load the FAISS index.\n\n"
        "Please build the vector database first before running the application."
    )
    st.stop()

# -------------------------------------------------------
# Header
# -------------------------------------------------------

st.title("📄 Potens RAG Pipeline")
st.caption("Research Paper Question Answering and Document Comparison")

tab1, tab2 = st.tabs(
    [
        "Ask Question",
        "Compare Documents"
    ]
)

# =======================================================
# QUESTION ANSWERING
# =======================================================

with tab1:

    st.subheader("Ask a Question")

    question = st.text_input(
        "Question",
        placeholder="Example: What methodology is proposed in the paper?"
    )

    if st.button("Ask", use_container_width=True):

        if question.strip():

            with st.spinner("Generating answer..."):

                result = answer_query(
                    vector_db,
                    question
                )

            st.markdown("## Answer")

            st.write(result["answer"])

            st.markdown("---")

            st.markdown("## Citations")

            citations = result.get("citations", [])

            if citations:

                for i, citation in enumerate(citations, start=1):

                    with st.expander(f"Citation {i}", expanded=True):

                        st.markdown(
                            f"""
**Source File:** {citation["source_file"]}

**Page:** {citation["page"]}

**Chunk ID:** {citation["chunk_id"]}

**Snippet**

> {citation["snippet"]}
"""
                        )

            else:
                st.info("No citations available.")

# =======================================================
# DOCUMENT COMPARISON
# =======================================================

with tab2:

    st.subheader("Compare Two Research Papers")

    pdf_files = []

    if os.path.exists(DOCUMENTS_FOLDER):

        pdf_files = sorted(
            [
                file
                for file in os.listdir(DOCUMENTS_FOLDER)
                if file.lower().endswith(".pdf")
            ]
        )

    if len(pdf_files) < 2:

        st.warning(
            "At least two PDF documents are required inside the Documents folder."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:
            document_a = st.selectbox(
                "Document A",
                pdf_files
            )

        with col2:
            default_index = 1 if len(pdf_files) > 1 else 0

            document_b = st.selectbox(
                "Document B",
                pdf_files,
                index=default_index
            )

        topic = st.text_input(
            "Comparison Topic",
            placeholder="Example: Model Performance"
        )

        if st.button("Compare", use_container_width=True):

            if topic.strip():

                with st.spinner("Comparing documents..."):

                    comparison = compare_documents(
                        vector_db,
                        os.path.join(DOCUMENTS_FOLDER, document_a),
                        os.path.join(DOCUMENTS_FOLDER, document_b),
                        topic
                    )

                st.markdown("## Verdict")

                verdict = comparison["verdict"]

                if verdict == "Agree":
                    st.success(verdict)

                elif verdict == "Partially Agree":
                    st.warning(verdict)

                elif verdict == "Contradict":
                    st.error(verdict)

                else:
                    st.info(verdict)

                st.markdown("## Reasoning")

                st.write(comparison["explanation"])

                st.markdown("---")

                col1, col2 = st.columns(2)

                # --------------------------------------
                # Evidence A
                # --------------------------------------

                with col1:

                    st.markdown("### Evidence from Document A")

                    evidence_a = comparison.get(
                        "evidence_from_document_a",
                        []
                    )

                    if evidence_a:

                        for evidence in evidence_a:

                            with st.expander(
                                f'{evidence["source_file"]} | Page {evidence["page"]}'
                            ):

                                st.markdown(
                                    f"""
**Chunk ID:** {evidence["chunk_id"]}

> {evidence["snippet"]}
"""
                                )

                    else:
                        st.info("No supporting evidence found.")

                # --------------------------------------
                # Evidence B
                # --------------------------------------

                with col2:

                    st.markdown("### Evidence from Document B")

                    evidence_b = comparison.get(
                        "evidence_from_document_b",
                        []
                    )

                    if evidence_b:

                        for evidence in evidence_b:

                            with st.expander(
                                f'{evidence["source_file"]} | Page {evidence["page"]}'
                            ):

                                st.markdown(
                                    f"""
**Chunk ID:** {evidence["chunk_id"]}

> {evidence["snippet"]}
"""
                                )

                    else:
                        st.info("No supporting evidence found.")