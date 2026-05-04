import os
from openai import OpenAI

# Option 1: Set your API key here
# os.environ["OPENAI_API_KEY"] = "your_api_key_here"

# Option 2: Read from .env (recommended)
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("❌ OPENAI_API_KEY not found in environment variables.")
    print("Please set it in your .env file or directly in the code.")
    exit(1)
try:
    client = OpenAI(api_key=openai_api_key)

    response = client.models.list()

    print("✅ API Key is VALID")
    print(f"Available models: {[model.id for model in response.data][:5]}")

except Exception as e:
    print("❌ API Key is INVALID or not working")
    print("Error:", str(e))