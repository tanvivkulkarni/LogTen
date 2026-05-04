from google import genai
from google.genai import types
from dotenv import load_dotenv
from pathlib import Path
import argparse
import os
import sys
import csv
import io
import re
from datetime import datetime

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "models/gemini-2.5-pro"

OCR_PROMPT = """You are a precision OCR engine. Your task is to extract **every single character** from this image exactly as it appears. Accuracy is critical — aim for 99%+ fidelity.

## Rules — follow strictly:

1. **Verbatim transcription**: Reproduce all text character-for-character. Do NOT correct spelling, grammar, punctuation, or formatting errors — they must be preserved exactly. Do NOT skip, merge, or summarize any content.
2. **Reading order**: Read left-to-right, top-to-bottom. For multi-column layouts, transcribe each column separately in order (left column first, then right).
3. **Structure preservation**:
   - Headings: prefix with `#`, `##`, `###` based on visual hierarchy.
   - Paragraphs: separate with a blank line.
   - Bullet / numbered lists: reproduce with `-` or the original numbering.
   - Tables — this is CRITICAL, follow exactly:
     a. Count the EXACT number of columns and rows in the image before you begin. State this count to yourself first.
     b. Render as Markdown tables with `|` delimiters.
     c. Every row in the image MUST appear as a row in your output — do NOT skip or merge rows.
     d. Every column in the image MUST appear as a column — do NOT skip, merge, or add extra columns.
     e. Preserve every cell value exactly as printed — numbers, decimals, commas, units, abbreviations, symbols.
     f. For empty cells, leave the cell blank between `|` delimiters with padding spaces to match column width (e.g., `|           |`).
     g. **Column alignment is MANDATORY**: Every column must have a FIXED width. Pad EVERY cell with trailing spaces so that all `|` delimiters line up vertically across all rows. Find the widest value in each column and pad all other cells in that column to match. Example of correctly padded output:
        ```
        | DEC.  | 8    | C152   | TLN    | SELF             |
        | DEC.  | 9    | C152   | TLN    | M. HACKWOOD      |
        ```
        NOT like this (wrong — no padding):
        ```
        | DEC. | 8 | C152 | TLN | SELF |
        | DEC. | 9 | C152 | TLN | M. HACKWOOD |
        ```
     h. Use solid dash separators with width matching the column (e.g., `|-------------|`) — do NOT use colon-alignment syntax (`:---`).
     i. Each cell should contain plain text only — no backticks, no inline code formatting, no bold/italic.
     j. If a table spans multiple pages or sections, keep it as one continuous table.
     k. After writing the table, re-examine the image and verify: (1) row count matches, (2) column count matches, (3) no cell values were swapped or misaligned.
4. **Special elements**:
   - Handwritten text: transcribe as-is, prefix uncertain words with `[?]`.
   - **Rubber stamps, certification blocks, and signatures** (including text printed on stamps that overlaps table cells, e.g. "I consider ... competent to fly", "solo by day/night", flying-school "correct to date" stamps, licence numbers printed on stamps, and illegible signatures): **DO NOT TRANSCRIBE ANY OF IT.** Treat overlapped table cells as **empty**. Logbook flight rows and printed column headers are data; stamp overlay text is **not** data.
   - Other watermarks, annotations, logos, and non-text graphics: DO NOT TRANSCRIBE THEM. Omit entirely.
   - **Bleed-through / show-through / reverse-side text** visible faintly from the back of the page is **NOT part of the current page**. Ignore it completely. If the front-side table cells are blank but faint mirrored or background text appears from the reverse side, those cells must stay blank in your output.
   - Checkboxes: `[x]` for checked, `[ ]` for unchecked.
   - Strikethrough text: wrap in `~~strikethrough~~`.
5. **Whitespace & alignment**: Preserve meaningful indentation and spacing. Collapse excessive whitespace to single spaces within a line.
6. **Symbols & special characters**: Reproduce math symbols, currency signs, accented characters, superscripts, subscripts exactly. Use Unicode where possible. Pay special attention to: `0` vs `O`, `1` vs `l` vs `I`, `5` vs `S`, `8` vs `B`, `,` vs `.` in numbers.
7. **Confidence**: If a character or word is ambiguous/illegible, give your best guess and mark it with `[?]`.
8. **Self-verification**: After completing the transcription, re-read the image one final time and compare against your output. Fix any discrepancies before responding.
9. **Output nothing else**: Do not add explanations, commentary, summaries, or metadata. Output ONLY the transcribed text.
"""


VERIFY_PROMPT = """You are a precision OCR verifier. Below is a transcription that was extracted from the attached image. Your job is to compare the transcription against the image and produce a CORRECTED version.

## Instructions:
1. Go through the image carefully, cell by cell, line by line.
2. Compare every character, number, and symbol in the transcription against what actually appears in the image.
3. Fix ANY errors you find: wrong numbers, missing rows, missing columns, swapped values, extra/missing text, wrong symbols.
4. For tables: verify the row count, column count, and that every cell value matches the image exactly.
5. For tables: ensure every column has a FIXED width with spaces padding each cell so all `|` delimiters align vertically. Every cell in a column must be the same width.
6. Output ONLY the corrected transcription. If the original was already perfect, output it unchanged.
6. Do NOT add any commentary, explanations, or notes about what you changed.
7. If the transcription contains any `[STAMP: ...]`, `[WATERMARK: ...]`, `[ANNOTATION: ...]` text, remove it entirely.
8. Remove any rubber-stamp or certification-overlay text and signatures from table cells (same rule as the initial OCR: those cells must be blank in the output).
9. Remove any bleed-through / show-through / reverse-side text. Faint background text from the back of the page is not part of the current page and must not be transcribed; blank front-side cells must remain blank.

## Transcription to verify:
{transcription}
"""


def _load_image(image_path: str):
    """Load image and return (bytes, mime_type)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

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
    mime_type = mime_map.get(path.suffix.lower(), "image/png")
    return path.read_bytes(), mime_type


def ocr_image(image_path: str, verify: bool = True) -> str:
    """Extract text from an image file using Gemini 2.5 Pro with optional verification pass."""
    image_bytes, mime_type = _load_image(image_path)

    # Pass 1: Initial OCR
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part.from_text(text=OCR_PROMPT),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=65536,
        ),
    )
    first_pass = response.text if isinstance(response.text, str) else ""

    if not verify:
        return first_pass

    # Pass 2: Verification — re-examine image and fix errors
    verify_text = VERIFY_PROMPT.format(transcription=first_pass)
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part.from_text(text=verify_text),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=65536,
        ),
    )
    return response.text if isinstance(response.text, str) else first_pass


def _is_markdown_table_line(line: str) -> bool:
    s = line.strip("\n")
    return s.strip().startswith("|") and "|" in s[1:]


def _is_separator_row(line: str) -> bool:
    # Matches rows like: |-----|----| with optional spaces
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")):
        return False
    inner = s[1:-1].strip()
    if not inner:
        return False
    parts = [p.strip() for p in inner.split("|")]
    if not parts:
        return False
    for p in parts:
        if not p:
            return False
        if any(ch not in "- " for ch in p):
            return False
        if "-" not in p:
            return False
    return True


def _parse_markdown_table_lines(table_lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in table_lines:
        if _is_separator_row(line):
            continue
        s = line.strip()
        if not (s.startswith("|") and s.endswith("|")):
            continue
        cells = [c.strip() for c in s[1:-1].split("|")]
        rows.append(cells)
    return rows


def _csv_text_from_rows(rows: list[list[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for r in rows:
        writer.writerow(r)
    return buf.getvalue().rstrip("\n")


def pad_rows_to_rectangular(rows: list[list[str]]) -> list[list[str]]:
    """Pad all rows with '' so every row has the same column count."""
    if not rows:
        return rows
    max_cols = max(len(r) for r in rows)
    if max_cols <= 0:
        return rows
    return [r + [""] * (max_cols - len(r)) for r in rows]


def extract_all_table_block_lines(text: str) -> list[list[str]]:
    """Return every Markdown table block found in the text."""
    lines = text.splitlines()
    blocks: list[list[str]] = []
    i = 0
    while i < len(lines):
        if _is_markdown_table_line(lines[i]):
            j = i
            while j < len(lines) and _is_markdown_table_line(lines[j]):
                j += 1
            block = lines[i:j]
            if any(_is_separator_row(x) for x in block) and len(block) >= 2:
                blocks.append(block)
            i = j
        else:
            i += 1
    return blocks


def extract_all_table_rows(text: str) -> list[list[list[str]]]:
    """Return parsed rows for every Markdown table block found in the text."""
    return [_parse_markdown_table_lines(block) for block in extract_all_table_block_lines(text)]


def extract_first_table_rows(text: str) -> list[list[str]]:
    """Return parsed rows (list of cell lists) from the first Markdown table block."""
    all_rows = extract_all_table_rows(text)
    return all_rows[0] if all_rows else []


def extract_first_table_block_lines(text: str) -> list[str]:
    """Return the raw Markdown lines for the first table block (including separator rows)."""
    all_blocks = extract_all_table_block_lines(text)
    return all_blocks[0] if all_blocks else []


def strip_stamps(text: str) -> str:
    """
    Remove any leftover stamp markers from model output.
    (Even with prompting, models occasionally emit them inside table cells.)
    """
    if text is None:
        return ""
    if not isinstance(text, (str, bytes)):
        text = str(text)
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return re.sub(r"\[(STAMP|WATERMARK|ANNOTATION)\s*:[^\]]*\]", "", text, flags=re.IGNORECASE)


def _cell_looks_like_stamp_or_signature(cell: str) -> bool:
    """Heuristic: logbook numeric cells rarely contain stamp boilerplate or signature tags."""
    if not cell or not str(cell).strip():
        return False
    raw = str(cell)
    t = " ".join(raw.split()).upper()
    tc = re.sub(r"\s+", "", t)

    if "[SIGNATURE]" in raw or "SIGNATURE]" in raw:
        return True

    stamp_phrases = (
        "FOGARTY",
        "AVIATIONACADEMY",
        "CORRECTTODATE",
        "CORRECT TO DATE",
        "ICONSIDER",
        "I CONSIDER",
        "CONSIDER",
        "COMPETENTTOFLY",
        "COMPETENT TO FLY",
        "SOLOBYDAY",
        "SOLO BY DAY",
        "SOLO BY NIGHT",
        "SOLO FLIGHT",
        "TYPEAIRCRAFT",
        "TYPE AIRCRAFT",
        "HE/SHEHASBEEN",
        "HE/SHE HAS BEEN",
        "INSTRUCTEDINANDFOUND",
        "ALLSEQUENCESFORTHIS",
        "ALL SEQUENCES FOR THIS",
        "COMPETENTINALL",
    )
    if any(p.replace(" ", "") in tc for p in stamp_phrases):
        # "CONSIDER" alone is weak; pair with stamp-like context
        if "CONSIDER" in t and ("COMPETENT" in t or "SOLO" in t or "FLY" in t):
            return True
        if "FOGARTY" in t or "AVIATIONACADEMY" in tc or "CORRECTTODATE" in tc:
            return True
        if "SOLOBYDAY" in tc or "SOLO BY DAY" in t:
            return True
        if "HE/SHE" in raw.upper() and "INSTRUCTED" in t:
            return True
        if "ALL SEQUENCES" in t or "ALLSEQUENCES" in tc:
            return True

    if re.search(r"\bNO\.?\s*(?:SCPL|CPL|SPL)\b", t, re.I):
        return True
    if re.search(r"\b(?:SCPL|CPL)\b", t) and len(raw) < 100:
        return True

    return False


def clean_stamp_cells_in_rows(rows: list[list[str]]) -> list[list[str]]:
    out: list[list[str]] = []
    for row in rows:
        out.append(["" if _cell_looks_like_stamp_or_signature(c) else c for c in row])
    return out


def clean_stamp_cells_in_markdown_table_lines(lines: list[str]) -> list[str]:
    """Strip stamp/signature text from inside | cells | while preserving layout."""
    cleaned: list[str] = []
    for line in lines:
        s = line.strip()
        if not (s.startswith("|") and s.endswith("|")):
            cleaned.append(line)
            continue
        parts = [c.strip() for c in s[1:-1].split("|")]
        parts = ["" if _cell_looks_like_stamp_or_signature(c) else c for c in parts]
        rebuilt = "|" + "|".join(parts) + "|"
        cleaned.append(rebuilt)
    return cleaned


def append_csv_below_first_table(text: str, only_csv_path: Path) -> str:
    """
    Find all Markdown table blocks, write their CSVs to only_csv_path,
    and inject each CSV immediately below its table in the returned text.
    """
    lines = text.splitlines()
    blocks = extract_all_table_block_lines(text)
    all_rows = extract_all_table_rows(text)
    cleaned_blocks: list[list[list[str]]] = []
    for rows in all_rows:
        rows = clean_stamp_cells_in_rows(rows)
        rows = pad_rows_to_rectangular(rows)
        if rows:
            cleaned_blocks.append(rows)
    if not blocks or not cleaned_blocks:
        return text

    only_csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_sections = [_csv_text_from_rows(rows) for rows in cleaned_blocks]
    with open(only_csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("\n\n".join(section for section in csv_sections if section))

    injected: list[str] = []
    i = 0
    block_index = 0
    while i < len(lines):
        if _is_markdown_table_line(lines[i]):
            j = i
            while j < len(lines) and _is_markdown_table_line(lines[j]):
                j += 1
            block = lines[i:j]
            if any(_is_separator_row(x) for x in block) and len(block) >= 2:
                injected.extend(block)
                if block_index < len(csv_sections) and csv_sections[block_index]:
                    injected.append("")
                    injected.extend(csv_sections[block_index].splitlines())
                block_index += 1
                i = j
                continue
        injected.append(lines[i])
        i += 1

    return "\n".join(injected)


def write_table_then_csv_as_valid_csv(
    out_path: Path, markdown_table_lines: list[str], table_rows: list[list[str]]
) -> None:
    """
    Write a single RFC4180-friendly CSV where:
      - the Markdown table is stored in column A (other columns blank)
      - a blank row
      - the parsed table rows as real CSV rows

    This keeps the file "real CSV" (constant column count per row) while still
    containing "table then CSV" content.
    """
    max_cols = max((len(r) for r in table_rows), default=1)
    if max_cols < 1:
        max_cols = 1

    def pad(row: list[str]) -> list[str]:
        if len(row) >= max_cols:
            return row
        return row + [""] * (max_cols - len(row))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")

        for line in markdown_table_lines:
            writer.writerow([line] + [""] * (max_cols - 1))

        # separator blank row
        writer.writerow([""] * max_cols)

        for r in table_rows:
            writer.writerow(pad(r))


def write_all_tables_then_csv_as_valid_csv(
    out_path: Path, markdown_table_blocks: list[list[str]], all_table_rows: list[list[list[str]]]
) -> None:
    """
    Write all tables as one valid CSV file:
    - each Markdown table block in column A
    - blank separator row
    - parsed CSV rows for that block
    - blank separator row between blocks
    """
    cleaned_rows: list[list[list[str]]] = []
    for rows in all_table_rows:
        rows = clean_stamp_cells_in_rows(rows)
        rows = pad_rows_to_rectangular(rows)
        if rows:
            cleaned_rows.append(rows)

    max_cols = 1
    for rows in cleaned_rows:
        max_cols = max(max_cols, max((len(r) for r in rows), default=1))

    def pad(row: list[str]) -> list[str]:
        if len(row) >= max_cols:
            return row
        return row + [""] * (max_cols - len(row))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        for idx, block in enumerate(markdown_table_blocks):
            for line in block:
                writer.writerow([line] + [""] * (max_cols - 1))
            writer.writerow([""] * max_cols)
            if idx < len(cleaned_rows):
                for row in cleaned_rows[idx]:
                    writer.writerow(pad(row))
            if idx != len(markdown_table_blocks) - 1:
                writer.writerow([""] * max_cols)


def _looks_like_valid_table_result(text: str) -> bool:
    """Heuristic: we have at least one Markdown table block with parsed rows."""
    if not text or not isinstance(text, str):
        return False
    md_blocks = extract_all_table_block_lines(text)
    if not md_blocks:
        return False
    all_rows = extract_all_table_rows(text)
    for rows in all_rows:
        rows = clean_stamp_cells_in_rows(rows)
        rows = pad_rows_to_rectangular(rows)
        if len(rows) >= 2 and any(any(cell.strip() for cell in r) for r in rows):
            return True
    return False


def ocr_with_retries(image_path: str, verify: bool, retries: int = 3) -> str:
    """
    Run OCR up to N times until we get a detectable table result.
    Returns the best attempt (first valid), or the last attempt.
    """
    last = ""
    for attempt in range(1, max(1, retries) + 1):
        try:
            text = ocr_image(image_path, verify=verify)
        except Exception as e:
            text = ""
            last = f""
            # keep looping
        text = strip_stamps(text)
        last = text
        if _looks_like_valid_table_result(text):
            return text
    return last


# def ocr_image_url(image_url: str) -> str:
#     """Extract text from an image URL using Gemini 2.5 Pro."""
#     response = client.models.generate_content(
#         model=MODEL,
#         contents=[
#             types.Content(
#                 role="user",
#                 parts=[
#                     types.Part.from_uri(file_uri=image_url, mime_type="image/png"),
#                     types.Part.from_text(text=OCR_PROMPT),
#                 ],
#             )
#         ],
#         config=types.GenerateContentConfig(
#             temperature=0.0,
#             top_p=1.0,
#             max_output_tokens=65536,
#         ),
#     )
#     return response.text

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def _collect_image_paths(root: Path) -> list[Path]:
    """Single image file, or all images in a directory (or under original/enhanced)."""
    root = root.resolve()
    if root.is_file():
        if root.suffix.lower() not in _IMAGE_SUFFIXES:
            return []
        return [root]
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
            out.append(p)
    if not out:
        for sub in ("original", "enhanced"):
            subdir = root / sub
            if subdir.is_dir():
                for p in sorted(subdir.iterdir()):
                    if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
                        out.append(p)
    return sorted(out)


def _resolve_batch_name(path: Path, override: str | None) -> str:
    """
    Subfolder under only_csv/ and with_tables/.
    --batch-name overrides; else inferred from path.
    """
    if override:
        return override.strip()
    p = path.resolve()
    if p.is_file():
        if p.parent.name.lower() in ("original", "enhanced") and p.parent.parent.name:
            return p.parent.parent.name
        return p.parent.name
    if p.name.lower() in ("original", "enhanced") and p.parent.name:
        return p.parent.name
    return p.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OCR logbook images with Gemini; write only_csv and with_tables outputs."
    )
    parser.add_argument(
        "path",
        help="One image or a folder of images (or .../batch with original/enhanced inside)",
    )
    parser.add_argument("--no-verify", action="store_true", help="Skip second verification pass")
    parser.add_argument(
        "--batch-name",
        default=None,
        help="Override output subfolder under only_csv/ and with_tables/ (default: inferred from path)",
    )
    parser.add_argument(
        "--only-csv-root",
        default="only_csv",
        help="Root directory for table-only CSV (default: only_csv)",
    )
    parser.add_argument(
        "--with-tables-root",
        default="with_tables",
        help="Root directory for markdown+CSV output (default: with_tables)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per image when OCR output is empty/no-table (default: 3)",
    )
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"Not found: {root}", file=sys.stderr)
        sys.exit(1)

    images = _collect_image_paths(root)
    if not images:
        print("No image files found (.png, .jpg, ...).", file=sys.stderr)
        sys.exit(1)

    batch_name = _resolve_batch_name(root, args.batch_name)
    do_verify = not args.no_verify
    only_root = Path(args.only_csv_root)
    tables_root = Path(args.with_tables_root)

    for image_path in images:
        target = str(image_path)
        stem = image_path.stem

        result = ocr_with_retries(target, verify=do_verify, retries=args.retries)
        only_csv_path = only_root / batch_name / f"gen_output_{stem}.csv"
        result_with_csv = append_csv_below_first_table(result, only_csv_path)

        print(result_with_csv)

        out_path = tables_root / batch_name / f"gen_output_{stem}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        md_blocks = extract_all_table_block_lines(result)
        all_rows = extract_all_table_rows(result)
        if md_blocks and all_rows:
            write_all_tables_then_csv_as_valid_csv(out_path, md_blocks, all_rows)
        else:
            # If we still failed after retries, write an explicit error instead of an empty CSV.
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(
                    [
                        "ERROR",
                        f"No table detected after {args.retries} attempt(s)",
                        target,
                    ]
                )

        log_path = Path("output_folder/csv_output/gem_runs.csv")
        log_exists = log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            if not log_exists:
                writer.writerow(["timestamp", "source", "output_file"])
            writer.writerow([datetime.now().isoformat(), target, str(out_path)])