import argparse
import os
from pathlib import Path


DEFAULT_MODEL = "models/gemini-2.5-flash"

DEFAULT_PROMPT = """You are a data extraction system.

Your task is to convert text into a structured CSV output where metadata appears first, followed by the table.

INPUT

The OCR text contains:

Metadata (top section)
A table with:
One header row separated by "|"
Multiple data rows separated by "|"
STEP 1: EXTRACT METADATA
Read all lines before the table begins
Convert metadata into key-value pairs
If a value contains commas (e.g., "abc, xyz"), treat it as a single value
Output each metadata item as:

Key,Value

Do not add extra spaces or empty lines
STEP 2: PROCESS TABLE
Identify the first "|" separated line as the table header
Replace "|" with "," to convert into CSV format
Keep the header text unchanged
Process all rows using the same structure
Preserve empty cells exactly
STEP 3: HANDLE COMMAS IN VALUES
If any field value contains a comma, wrap the entire value in double quotes
Example:
Input: High quality, durable
Output: "High quality, durable"
Do not add quotes if there is no comma in the value
STEP 4: ALIGN METADATA WITH TABLE
Ensure metadata rows have the same number of columns as the table
Add empty values where needed to match the column count
STEP 5: OUTPUT STRUCTURE

Output in the following exact order:

Metadata rows (Key,Value + empty columns to match table width)
Table header row
All table data rows
RULES
Use "," only as the column separator in the final output
Do not use "|" in the final output
Do not add explanations or extra text
Do not wrap output in code blocks or markers
Do not merge metadata into table rows
Do not repeat metadata
Do not modify or reorder table headers
Do not skip any rows
Remove unnecessary empty lines
Ensure consistent column alignment across all rows
Preserve original text exactly except for required formatting changes
"""


def _call_gemini(text: str, *, model: str, prompt: str) -> str:
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment (.env).")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_text(text="\n\n=== INPUT TEXT ===\n" + text),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=65536,
        ),
    )
    return resp.text if isinstance(resp.text, str) else ""


def _load_bytes_and_mime(path: Path) -> tuple[bytes, str]:
    data = path.read_bytes()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".pdf": "application/pdf",
    }
    return data, mime_map.get(path.suffix.lower(), "application/octet-stream")


def _call_gemini_with_header_image(
    text: str, *, header_image: Path, model: str, prompt: str
) -> str:
    """
    Same as `_call_gemini`, but also provides a header image (column names/table header)
    to help the model produce better column alignment and naming.
    """
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment (.env).")

    client = genai.Client(api_key=api_key)
    img_bytes, img_mime = _load_bytes_and_mime(header_image)

    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=img_bytes, mime_type=img_mime),
                    types.Part.from_text(
                        text="This image shows the table header/column labels. Use it as the source of truth for column names and order."
                    ),
                    types.Part.from_text(text=prompt),
                    types.Part.from_text(text="\n\n=== INPUT TEXT ===\n" + text),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=65536,
        ),
    )
    return resp.text if isinstance(resp.text, str) else ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert extracted formatted text to CSV using a Gemini model."
    )
    parser.add_argument("input", help="Path to extracted text file (.txt)")
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Path to output CSV (default: <input_stem>.csv next to input)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f'Gemini model name (default: "{DEFAULT_MODEL}")',
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional path to a .txt file containing the prompt to use",
    )
    parser.add_argument(
        "--header-image",
        default=None,
        help="Optional path to a header image (e.g. .../header_image.png) to improve column naming/alignment",
    )
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    out_path = Path(args.out).resolve() if args.out else in_path.with_suffix(".csv")

    raw_text = in_path.read_text(encoding="utf-8", errors="replace")
    prompt = (
        Path(args.prompt_file).read_text(encoding="utf-8", errors="replace")
        if args.prompt_file
        else DEFAULT_PROMPT
    )

    if args.header_image:
        header_image = Path(args.header_image).resolve()
        if not header_image.exists():
            raise SystemExit(f"Header image not found: {header_image}")
        csv_text = _call_gemini_with_header_image(
            raw_text, header_image=header_image, model=args.model, prompt=prompt
        ).strip()
    else:
        csv_text = _call_gemini(raw_text, model=args.model, prompt=prompt).strip()
    if not csv_text:
        raise SystemExit("Gemini returned empty output.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(csv_text + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

