from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(directory: str = "documents") -> List[Document]:
    documents = []

    for file_path in Path(directory).glob("*"):

        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        try:

            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))

            elif suffix == ".docx":
                loader = Docx2txtLoader(str(file_path))

            elif suffix == ".txt":
                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8",
                )

            else:
                continue

            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file_path.name

            documents.extend(docs)

        except Exception as error:
            print(
                f"Failed to load {file_path}: {error}"
            )

    return documents


def split_documents(
    documents: List[Document],
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    return splitter.split_documents(documents)