import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

print("Available Embedding Models:")
print("-" * 50)
for m in client.models.list():
    if 'embedContent' in getattr(m, 'supported_actions', []):
        print(f"Name: {m.name}")
print("-" * 50)
