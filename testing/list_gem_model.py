from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
genai_api_token = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=genai_api_token)

print("List of models that support generateContent:\n")
for m in client.models.list():
    for action in m.supported_actions:
        if action == "generateContent":
            print(m.name)

print("List of models that support embedContent:\n")
for m in client.models.list():
    for action in m.supported_actions:
        if action == "embedContent":
            print(m.name)