import sys
import os
from dotenv import load_dotenv

# Add the project root to sys.path to allow running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from app.custom_model.gemini import GeminiPreviewEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

persistent_db_dir = "./.chroma_db"
db = Chroma(
    persist_directory=persistent_db_dir,
    embedding_function=GeminiPreviewEmbeddings(),
    collection_metadata={"hnsw:space": "cosine"}
)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GEMINI_API_KEY")  
)

chat_history = []

def ask_question(query):
    search_query = query
    
    if chat_history:
        # Create a prompt to rewrite the query based on history
        rewrite_messages = [
            SystemMessage(content="Given the chat history, rewrite the user's new query to be a standalone, searchable question. Just return the rewritten query text."),
        ]
        # Add history to the rewrite prompt
        rewrite_messages.extend(chat_history)
        # Add current query
        rewrite_messages.append(HumanMessage(content=query))

        search_query = llm.invoke(rewrite_messages).content.strip()
        print(f"(Rewritten Query: {search_query})")

    retrieved_docs = retriever.invoke(search_query)
    context = "\n".join([f"Source: {doc.metadata.get('source')}\nContent: {doc.page_content}" for doc in retrieved_docs])
    
    combined_input = f"""Based on the following context, answer the query: "{query}"

    Context:
    {context}

    Please provide a clear and concise answer. 
    If you cannot find the answer in the provided documents, please say i dont have enough information to answer the query.
    """
    
    answer_messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents."),
        HumanMessage(content=combined_input)
    ]
    
    response = llm.invoke(answer_messages)
    
    # Store history for future turns
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response.content))
    
    return response.content


def start_chat():
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        response = ask_question(query)
        print("Bot:", response)

start_chat()