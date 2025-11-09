import pandas as pd
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.retrievers import MultiQueryRetriever
from dotenv import load_dotenv


load_dotenv()
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")

model = OpenAIEmbeddings(
    api_key=OPEN_AI_KEY,
    model = "text-embedding-3-large"
)

llm = ChatOpenAI(
    model = 'gpt-4o-mini',
    api_key= OPEN_AI_KEY
)

vectorstore = FAISS.load_local(
    "faiss_shl_db3",
    model,
    allow_dangerous_deserialization=True
)

base_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "lambda_mult": 0.4}
)

retriever = MultiQueryRetriever.from_llm(
    llm=llm,
    retriever=base_retriever
)

prompt = ChatPromptTemplate.from_template("""
You are an SHL recommendation engine.

Given:
User query: {input}

Retrieved Context:
{context}

TASK:
From the retrieved documents, select the top 5 most relevant SHL assessments.
For each assessment, return a Python dictionary with EXACTLY these keys:

[
  {{
    "Assessment Name": "",
    "Job Levels": "",
    "Description": "",
    "Language": "",
    "Assessment Length": "",
    "Test Type": "",
    "URL": ""
  }}
]

RULES:
- Fill values ONLY from the retrieved context.
- If a field is missing, return an empty string.
- Output ONLY a list of 5 Python dictionaries.
- NO markdown, NO backticks, NO explanation.
""")


# Combine docs + LLM
stuff_chain = create_stuff_documents_chain(llm, prompt)

# Final RAG chain
rag_chain = create_retrieval_chain(retriever, stuff_chain)

resp = rag_chain.invoke({"input": "need a test for banking sales role"})
print(resp["answer"])