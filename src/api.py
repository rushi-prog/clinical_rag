from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_service import ask
app = FastAPI()


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():

    return {
        "message": "Clinical Trial RAG API"
    }


@app.post("/ask")
def ask_question(
    request: QueryRequest
):

    query = request.question

    # later connect retriever

    result = ask(query)

    return {
        "question": query,
        "answer": result["answer"],
        "sources": result["sources"]
    }