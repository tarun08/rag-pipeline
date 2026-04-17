import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

print("Available Models:")
print("-" * 50)
for m in client.models.list():
    print(f"Name: {m.name}")
    # Supported actions usually contains 'generateContent', 'embedContent', etc.
    actions = getattr(m, 'supported_actions', [])
    print(f"Supported Actions: {', '.join(actions) if actions else 'Unknown'}")
    print("-" * 50)
