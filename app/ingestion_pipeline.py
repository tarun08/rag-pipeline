from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import time
from app.custom_model.gemini import GeminiPreviewEmbeddings

load_dotenv()

def load_docs():
    if not os.path.exists("docs"):
        raise FileNotFoundError("No documents found in the ./docs directory")

    loader = DirectoryLoader("docs", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=100):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    all_chunks = []
    for doc in documents:
        # Split each document individually and take the first 10 chunks
        doc_chunks = text_splitter.split_documents([doc])
        all_chunks.extend(doc_chunks[:10])
        print(f"Extracted {len(doc_chunks[:10])} chunks from {doc.metadata.get('source')}")

    if all_chunks:
        print(f"Total chunks to embed: {len(all_chunks)}")
        for i, chunk in enumerate(all_chunks[:3]): # Preview first 3
            print(f"Preview Chunk {i+1} from {chunk.metadata['source']}")
    else:
        print("No documents found")

    return all_chunks


def create_vectorstore(texts):
    api_key = os.getenv("GEMINI_API_KEY")

    embeddings = GeminiPreviewEmbeddings()

    # ensure valid docs only
    texts = [t for t in texts if t.page_content and t.page_content.strip()]

    print(f"Embedding {len(texts)} valid chunks")

    try:
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory="./.chroma_db"
        )
        
        batch_size = 5 # Small batch size for rate limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")
            vectorstore.add_documents(documents=batch)
            if i + batch_size < len(texts):
                print("Sleeping for 10 seconds to respect rate limits...")
                time.sleep(10)

        print("Vectorstore created successfully!")
        return vectorstore

    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        return None

def process_docs():
    documents = load_docs()
    texts = split_documents(documents)
    
    # print(f"Starting full ingestion: Processing {len(texts)} chunks")
    
    # # vectorstore = create_vectorstore(texts)

    # print("Ingestion complete!")
