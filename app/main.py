from fastapi import FastAPI
from pydantic import BaseModel

from app.rag.chatbot import ITSupportChatbot


app = FastAPI(
    title="IT Support RAG Chatbot",
    version="1.0.0",
)


chatbot = ITSupportChatbot()


class Question(BaseModel):
    question: str


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "IT Support RAG Chatbot",
    }


@app.post("/ask")
def ask_question(request: Question):
    return chatbot.ask(request.question)