from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.rag.loader import load_documents, split_documents


def main():

    print("Loading IT documents...")

    documents = load_documents("documents")

    print(f"Loaded {len(documents)} document(s).")

    if not documents:
        print(
            "No supported documents found "
            "inside the documents folder."
        )
        return

    chunks = split_documents(documents)

    print(f"Created {len(chunks)} document chunks.")

    print("Creating local FREE embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating Chroma vector database...")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db",
    )

    print(
        "IT Support knowledge base created successfully."
    )


if __name__ == "__main__":
    main()