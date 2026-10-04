from dotenv import load_dotenv

load_dotenv()

import re

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


class ITSupportChatbot:

    def __init__(self):

        # FREE local embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Chroma vector database
        self.vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings,
        )

        # Google Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite"
        )

        # Strict RAG prompt
        self.prompt = ChatPromptTemplate.from_template(
            """
You are an IT Support Assistant.

You must answer ONLY from the IT DOCUMENTATION below.

STRICT RULES:

1. Never use information from your general knowledge.

2. Every factual statement in your answer must be supported
   directly by the IT DOCUMENTATION.

3. NEVER invent, guess, complete, or infer:
   - URLs
   - website addresses
   - commands
   - procedures
   - policies
   - software names
   - company names
   - product names
   - troubleshooting steps

4. If a URL is not explicitly present in the documentation,
   DO NOT output any URL.

5. If the documentation says "company password portal",
   use exactly "company password portal".
   Do not replace it with a specific website.

6. Do not add steps that are not present in the documentation.

7. Preserve the meaning and order of documented procedures.

8. If the documentation does not contain enough information
   to answer the question, respond exactly:

I don't have enough information in the IT documentation to answer that.

9. At the end, identify the source document.

IT DOCUMENTATION:

{context}

USER QUESTION:

{question}

ANSWER:
"""
        )

    def ask(self, question: str):

        # Retrieve relevant documents
        documents = self.vectorstore.similarity_search(
            question,
            k=4,
        )

        if not documents:
            return {
                "answer": (
                    "I don't have enough information "
                    "in the IT documentation to answer that."
                ),
                "sources": [],
            }

        # Build documentation context
        context = "\n\n".join(
            [
                f"Source: {doc.metadata.get('source')}\n"
                f"{doc.page_content}"
                for doc in documents
            ]
        )

        # Build prompt
        messages = self.prompt.format_messages(
            context=context,
            question=question,
        )

        # Ask Gemini
        response = self.llm.invoke(messages)

        # Extract response text
        if isinstance(response.content, str):
            answer = response.content
        else:
            answer = "\n".join(
                item.get("text", "")
                for item in response.content
                if isinstance(item, dict)
                and item.get("type") == "text"
            )

        # --------------------------------------------------
        # SECURITY CHECK:
        # Remove URLs that were not present in documentation
        # --------------------------------------------------

        documentation_urls = set(
            re.findall(
                r"https?://[^\s<>\"]+",
                context
            )
        )

        answer_urls = re.findall(
            r"https?://[^\s<>\"]+",
            answer
        )

        for url in answer_urls:
            clean_url = url.rstrip(".,!?;:)")

            if clean_url not in documentation_urls:
                answer = answer.replace(url, "")

        # Sources
        sources = list(
            {
                doc.metadata.get("source")
                for doc in documents
                if doc.metadata.get("source")
            }
        )

        return {
            "answer": answer.strip(),
            "sources": sources,
        }