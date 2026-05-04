""" 
Handwritten text extraction using OpenAI Vision API.

Requirements (requirements.txt):
    openai>=1.14.0
    python-dotenv>=1.0.0

Python version: 3.9+
"""

import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OUTPUT_FILE = "gpt_5.4_output3.csv"

SYSTEM_PROMPT = (
    "You are a highly precise OCR engine. Your sole task is to transcribe every character "
    "visible in the image with 100% fidelity.\n\n"
    "## Rules\n"
    "1. Transcribe ALL text exactly as written — preserve original spelling, capitalization, "
    "punctuation, accents, abbreviations, strikethroughs, superscripts, and subscripts.\n"
    "2. Maintain the exact spatial layout: line breaks, indentation, paragraph spacing, "
    "and alignment must mirror the source.\n"
    "3. For tables: reproduce the structure using | for column separators and - for row "
    "separators. Preserve every header, row, column, and cell value exactly. Align columns "
    "so they are visually readable.\n"
    "4. For handwritten text: read each word carefully, paying close attention to letter shapes. "
    "Distinguish similar characters (e.g., 'l' vs '1', 'O' vs '0', 'rn' vs 'm', 'c' vs 'e', "
    "'u' vs 'v', 'a' vs 'o'). When uncertain, use surrounding context and word patterns to "
    "determine the most probable reading.\n"
    "5. Preserve numbered lists, bullet points, headings, underlines, and any structural "
    "formatting using plain-text equivalents.\n"
    "6. If text appears in multiple columns, transcribe left column first, then right column, "
    "separated by a blank line.\n"
    "7. Do NOT add any explanations, commentary, headers, footers, or metadata.\n"
    "8. Do NOT skip, summarize, or paraphrase any content.\n"
    "9. Do NOT mention uncertainty or illegibility — always output your best interpretation.\n"
    "10. Output ONLY the transcribed text, nothing else."
)


def encode_image(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = path.suffix.lower()
    supported = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    if suffix not in supported:
        raise ValueError(f"Unsupported image format '{suffix}'. Use one of: {supported}")

    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def extract_handwritten_text(image_path: str, model: str = "gpt-5.4") -> str:
    """Send the image to the OpenAI Vision API and return the extracted text."""
    b64_image = encode_image(image_path)

    suffix = Path(image_path).suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    if suffix == ".gif":
        media_type = "image/gif"
    elif suffix == ".webp":
        media_type = "image/webp"
        
    openai_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_image}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract all text from this image. "
                            "If it contains a table, reproduce the table structure exactly "
                            "using plain-text formatting with | and - characters."
                        ),
                    },
                ],
            },
        ],
        temperature=0,
        max_completion_tokens=2000,
    )

    return response.choices[0].message.content


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        text = extract_handwritten_text(image_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"API error: {e}")
        sys.exit(1)

    output_path = Path(OUTPUT_FILE)
    output_path.write_text(text, encoding="utf-8")

    print(f"\n--- Extracted Text (saved to {OUTPUT_FILE}) ---\n")
    # print(text)

if __name__ == "__main__":
    main()