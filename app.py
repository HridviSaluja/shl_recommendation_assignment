from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import MultiQueryRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import uvicorn
import ast
import re

load_dotenv()
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")

app = FastAPI(
    title="SHL RAG Server",
    description="AI-powered SHL assessment recommendation engine",
    version="1.0"
)

class QueryRequest(BaseModel):
    query: str

# LLM + Embeddings
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPEN_AI_KEY,
    temperature=0
)

embedding = OpenAIEmbeddings(
    api_key=OPEN_AI_KEY,
    model="text-embedding-3-large"
)

# Load FAISS DB
vectorstore = FAISS.load_local(
    "C:/Users/HP/Desktop/SHL/faiss_shl_db3",
    embedding,
    allow_dangerous_deserialization=True
)

base_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "lambda_mult": 0.4}
)

retriever = MultiQueryRetriever.from_llm(
    llm=llm,
    retriever=base_retriever
)

prompt = ChatPromptTemplate.from_template("""
You are an SHL recommendation engine.

User Query: {input}

Retrieved Context:
{context}

Return a list of EXACTLY 5 Python dictionaries, each containing:

{{
  "Assessment Name": "",
  "Job Levels": "",
  "Description": "",
  "Language": "",
  "Assessment Length": "",
  "Test Type": "",
  "URL": ""
}}

Rules:
- Use ONLY information from the retrieved context.
- No markdown, no backticks.
- Output ONLY the Python list.
""")

TEST_TYPE_MAPPING = {
    "A": "Ability and Aptitude",
    "B": "Biodata and Situational Judgement",
    "C": "Competencies, Development and 360",
    "E": "Assessment Exercise",
    "K": "Knowledge and Skills",
    "P": "Personality and Behavior",
    "S": "Simulations"
}

stuff_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, stuff_chain)

TEST_TYPE_MAPPING = {
    "A": "Ability and Aptitude",
    "B": "Biodata and Situational Judgement",
    "C": "Competencies, Development and 360",
    "E": "Assessment Exercise",
    "K": "Knowledge and Skills",
    "P": "Personality and Behavior",
    "S": "Simulations"
}

@app.post("/recommend")
def recommend(payload: QueryRequest):
    result = rag_chain.invoke({"input": payload.query})
    return {"recommendations": result["answer"]}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)