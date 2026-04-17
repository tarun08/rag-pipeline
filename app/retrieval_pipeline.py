import sys
import os
from dotenv import load_dotenv

# Add the project root to sys.path to allow running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from app.custom_model.gemini import GeminiPreviewEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

db  = Chroma(
    persist_directory="./.chroma_db",
    embedding_function=GeminiPreviewEmbeddings(),
    collection_metadata={"hnsw:space": "cosine"}
)

query = "what are achivements of isro?"

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

retrieved_docs = retriever.invoke(query)

print("\n" + "="*50)
print(f"QUERY: {query}")
print("="*50 + "\n")

if not retrieved_docs:
    print("No relevant documents found.")
else:
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "Unknown Source")
        print(f"[{i}] SOURCE: {source}")
        print("-" * 30)
        print(f"{doc.page_content.strip()}")
        print("-" * 50 + "\n")

# Prepare context from retrieved documents
context = "\n".join([f"Source: {doc.metadata.get('source')}\nContent: {doc.page_content}" for doc in retrieved_docs])

combined_input = f"""Based on the following context, answer the query: "{query}"

Context:
{context}

Please provide a clear and concise answer. 
If you cannot find the answer in the provided documents, please say i dont have enough information to answer the query.
"""

print("Context to LLM:")
print("-" * 50)
print(combined_input)
print("-" * 50 + "\n")

# Initialize Chat Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GEMINI_API_KEY")
)

messages = [
    SystemMessage(content="You are a helpful assistant that answers questions based on provided documents."),
    HumanMessage(content=combined_input)
]

print("Generating Response...")
response = llm.invoke(messages)

print("\nGenerated Response")
print("="*50 + "\n")

print(response.content)

print("\n" + "="*50)
print("END OF RESPONSE")
print("="*50)