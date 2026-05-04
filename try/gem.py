# from google import genai
# from dotenv import load_dotenv
# import os

# load_dotenv()
# google_api_key = os.getenv("GOOGLE_API_KEY")
# client = genai.Client(api_key=google_api_key)

# response = client.models.generate_content(
#     model="models/gemini-2.5-flash",
#     contents="Find the race condition in this multi-threaded C++ snippet: [code here]",
# )
# print(response.text)


from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
print(f"Using API key: {google_api_key[:4]}...{google_api_key[-4:]}")

client = genai.Client(api_key=google_api_key)

# Read image
with open("images/LB1 page 5_page-0001.jpg", "rb") as f:
    image_bytes = f.read()

response = client.models.generate_content(
    model="models/gemini-2.5-flash",
    contents=[
        {
            "role": "user",
            "parts": [
                {
                    "text": """Extract the text from this image and return it as a CSV table.

IMPORTANT RULES:
- Do NOT change any text
- Preserve spaces exactly
- Keep quotes (' and ") exactly as is
- Maintain column alignment
- Output ONLY raw CSV
"""
                },
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",  # ✅ FIXED
                        "data": image_bytes
                    }
                }
            ]
        }
    ]
)

# Write CSV
with open("output3.csv", "w", encoding="utf-8", newline="") as f:
    f.write(response.text.strip())

print("✅ CSV file saved as output.csv")