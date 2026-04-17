from google import genai
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
import os

load_dotenv()

class GeminiPreviewEmbeddings(Embeddings):
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def embed_documents(self, texts):
        return [
            self.client.models.embed_content(
                model="models/gemini-embedding-2-preview",
                contents=t
            ).embeddings[0].values
            for t in texts
        ]

    def embed_query(self, text):
        return self.client.models.embed_content(
            model="models/gemini-embedding-2-preview",
            contents=text
        ).embeddings[0].values