from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=google_api_key)

for m in client.models.list():
    print(m.name)