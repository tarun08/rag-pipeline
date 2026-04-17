from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

import os


def process_docs():
    documents = load_docs()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma.from_texts(texts, embeddings, persist_directory="./.chroma_db")
    print("Ingestion complete!")


def load_docs():
    if not os.path.exists("docs"):
        raise FileNotFoundError("No documents found in the ./docs directory")

    loader = DirectoryLoader("docs", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    return documents
