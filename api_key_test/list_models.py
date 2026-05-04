"""List all available OpenAI models for the configured API key."""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("OPENAI_API_KEY not found in environment variables.")
    exit(1)

try:
    client = OpenAI(api_key=openai_api_key)
    response = client.models.list()
    models = sorted([model.id for model in response.data])

    print(f"Available models ({len(models)}):\n")
    for model in models:
        print(f"  {model}")

except Exception as e:
    print(f"Error: {e}")
