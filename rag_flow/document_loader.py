import os

from langchain_community.document_loaders import PyPDFLoader


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
