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
     a. Count the EXACT number of columns and rows in the image before you begin.
     b. Render as Markdown tables with `|` delimiters.
     c. Every row in the image MUST appear as a row in your output — do NOT skip or merge rows.
     d. Every column in the image MUST appear as a column — do NOT skip, merge, or add extra columns.
     e. Preserve every cell value exactly as printed — numbers, decimals, commas, units, abbreviations, symbols.
     
     **EXCEPTION — DATE COLUMN NORMALIZATION (MANDATORY)**:
     - If the table contains a **Date column** OR separate **Month + Day columns**, you MUST ensure every row has a **complete date**.
     - Logbooks often use ditto marks (`"`, `''`, or blank cells). Handle them as follows:
       
       1. Identify the most recent **valid full date** above.
       2. If only a **day value** is present (e.g., `3rd`, `5th`, `8th`):
          - Extract numeric day → `03`, `05`, `08`
          - Combine with known month + year → `DD/MM/YYYY`
       3. If the cell contains ditto marks or is blank:
          - Repeat the previous full date.
       4. If Month is written once (e.g., DECEMBER 1985):
          - Apply it to all subsequent rows until changed.
       5. Output format MUST be strictly:
          → `DD/MM/YYYY`
       6. NEVER leave a Date cell empty.

     f. For empty cells (non-date columns), leave the cell blank between `|` delimiters with padding spaces.
     g. **Column alignment is MANDATORY**: Every column must have a FIXED width. Pad EVERY cell with trailing spaces so that all `|` delimiters align vertically across all rows.
     h. Use solid dash separators with width matching the column (e.g., `|-------------|`).
     i. Each cell should contain plain text only.
     j. If a table spans multiple pages or sections, keep it as one continuous table.
     k. After writing the table, verify:
        - Row count matches
        - Column count matches
        - No misalignment

4. **Special elements**:
   - Handwritten text: transcribe as-is, prefix uncertain words with `[?]`.
   - **Rubber stamps, certification blocks, signatures**: DO NOT TRANSCRIBE.
   - Watermarks, annotations, logos: OMIT.
   - **Bleed-through text**: IGNORE completely.
   - Checkboxes: `[x]` or `[ ]`
   - Strikethrough: `~~text~~`

5. **Whitespace & alignment**:
   - Preserve meaningful spacing.
   - Collapse excessive whitespace within lines.

6. **Symbols & characters**:
   - Preserve exactly (0/O, 1/I/l, punctuation, time formats).

7. **Confidence**:
   - Use `[?]` for uncertain text.

8. **Self-verification**:
   - Ensure ALL rows have valid dates.
   - Ensure dates follow correct chronological logic.
   - Ensure no row is skipped.

9. **Output nothing else**:
   - No explanations, only transcription.

10. **NO PROGRESSIVE / CUMULATIVE TOTALS (MANDATORY)**:
- Flight time values must be captured **exactly as written in each row ONLY**.
- DO NOT calculate, infer, or propagate running totals.
- DO NOT interpret any column as cumulative unless it is explicitly labeled as "Total" in the header.
- If a "TRIP TOTALS" or similar column exists:
  - Transcribe the value exactly as written.
  - DO NOT use it to modify or influence other rows.
- Each row must remain **independent**.

11. **TRIP TOTALS COLUMN — STRICT TRANSCRIPTION (DO NOT ALTER)**:
- The "Trip Totals" column MUST be copied EXACTLY as written in the image.
- Do the EXACAT Values of the trip totals column and remove the '.' decimal point instead use ':' semicolan.
- DO NOT:
  - Recalculate values
  - Normalize time formats
  - Fix or correct totals
  - Compare with other columns
  - Interpret as cumulative or non-cumulative
- Treat Trip Totals as a **visual field only**, not a computed field.
- Even if values appear inconsistent or incorrect, they MUST be preserved exactly.
"""


VERIFY_PROMPT = """You are a precision OCR verifier. Below is a transcription that was extracted from the attached image. Your job is to compare the transcription against the image and produce a CORRECTED version.

## Instructions:
1. Compare every character carefully against the image.
2. Fix ALL errors:
   - Wrong numbers
   - Missing rows/columns
   - Misaligned cells
   - Incorrect symbols

3. **Date Column Verification (CRITICAL)**:
   - Ensure EVERY row has a valid full date in `DD/MM/YYYY`.
   - If dates are missing, incomplete, or repeated incorrectly:
     a. Reconstruct using:
        - Day values in the row
        - Month/year from earlier rows
     b. Handle ditto marks or blanks by propagating previous date.
     c. Convert ordinal days:
        - `3rd` → `03`
        - `8th` → `08`
     d. Ensure chronological consistency.
     e. NEVER allow empty date cells.

4. **Tables**:
   - Verify row count and column count.
   - Ensure FIXED column width with proper padding.
   - Ensure perfect vertical alignment of `|`.

5. Output ONLY the corrected transcription.

6. Remove:
   - `[STAMP: ...]`, `[WATERMARK: ...]`, `[ANNOTATION: ...]`
   - Any stamp or signature overlays
   - Any bleed-through text

7. **Remarks section normalization (STRICT)**:
   Replace ONLY in Remarks column:

   - Circuits & Landings OR Circuits → C & L
   - Revision → REV
   - Instrument flight → I/F
   - Short Field Take Off & Landing → S.F.T.O & L
   - Training Area → T/A
   - Restricted Private Pilot’s License → RPPL
   - Constant Speed Unit → CSU.
   - Retractable Undercarriage → RT
   - Basic Instrument Flying → BIF
   - Maroochydore → MC
   - Brisbane → BN
   - Gayndah → GAY
   - Archerfield → AF
   - Maryborough → MYB
   - Short Field Takeoff and Landing → STOL

   Additional correction:
   - CYL / C Y L / Cyl → C & L (ONLY if context matches)

   Rules:
   - Case-insensitive match
   - Replace full phrases only
   - Do NOT modify outside Remarks column

8. **NO PROGRESSIVE TOTAL CORRECTION (CRITICAL)**:
   - Ensure that flight times are NOT treated as cumulative values.
   - Each row must reflect ONLY the value written in that row.
   - If any value appears to be incorrectly accumulated (e.g., increasing totals across rows),
     correct it by reverting to the actual per-row value from the image.
   - Do NOT recompute totals.
   - Do NOT propagate totals downward.
   - DO NOT CONSIDER Progressive Totals Carried Forward
   - DO NOT CONSIDER Progressive Totals
   - DO NOT CONSIDER Grand Total
   - DO NOT CONSIDER Total Aeronautical 


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


_MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "SEPT": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def _detect_year_from_ocr_text(text: str) -> int | None:
    if not text or not isinstance(text, str):
        return None
    m = re.search(r"\bYear[.\s]+(\d{4})\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        y = int(m.group(1))
    except ValueError:
        return None
    if 1900 <= y <= 2100:
        return y
    return None


def _format_date(year: int, month: int, day: int, date_format: str) -> str:
    fmt = (date_format or "DD/MM/YYYY").strip().upper()
    if fmt in ("YYYY-MM-DD", "YYYY/MM/DD"):
        sep = "-" if "-" in (date_format or "") else "-"
        return f"{year:04d}{sep}{month:02d}{sep}{day:02d}"
    if fmt in ("MM/DD/YYYY", "MM-DD-YYYY"):
        sep = "/" if "/" in fmt else "-"
        return f"{month:02d}{sep}{day:02d}{sep}{year:04d}"
    return f"{day:02d}/{month:02d}/{year:04d}"


def _parse_combined_month_day(cell: str) -> tuple[int | None, int | None]:
    """Parse a cell containing 'MONTH DAY' like 'JAN 27', 'JULY 28', 'DEC. 5'."""
    if not cell:
        return None, None
    s = cell.strip().upper()
    m = re.match(r"^([A-Z]+)\.?\s+(\d{1,2})$", s)
    if not m:
        return None, None
    mon_str, day_str = m.group(1), m.group(2)
    month = _MONTH_MAP.get(mon_str) or _MONTH_MAP.get(mon_str[:3])
    if month is None:
        return None, None
    try:
        day = int(day_str)
    except ValueError:
        return None, None
    if 1 <= month <= 12 and 1 <= day <= 31:
        return month, day
    return None, None


def _normalize_time_value(value: str, time_format: str) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""

    digits = re.sub(r"\D", "", s)
    if len(digits) < 3:
        return s
    if len(digits) == 3:
        digits = "0" + digits
    if len(digits) >= 4:
        digits = digits[:4]

    try:
        hh = int(digits[:2])
        mm = int(digits[2:4])
    except ValueError:
        return s
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return s

    tf = (time_format or "HHMM").strip().upper()
    if tf in ("HH:MM", "HHMM_WITH_COLON"):
        return f"{hh:02d}:{mm:02d}"
    return f"{hh:02d}{mm:02d}"


def _row_looks_like_header(row: list[str]) -> bool:
    """Heuristic: does this row contain Month and Day cells (i.e. it's a header row)?"""
    lc = [re.sub(r"\s+", " ", (c or "").strip()).lower() for c in row]
    return "month" in lc and "day" in lc


def _row_is_totals_or_stamp(row: list[str]) -> bool:
    """Return True for rows that are totals labels, stamp blocks, or blank separators."""
    joined = " ".join(c.strip() for c in row if c.strip()).lower()
    if not joined:
        return True
    totals_patterns = (
        "progressive totals",
        "flying times with",
        "bundy air charter",
        "to date",
        "grand total",
        "aeronautical experience",
    )
    return any(p in joined for p in totals_patterns)


def merge_table_blocks_for_page(all_block_rows: list[list[list[str]]]) -> list[list[str]]:
    """
    Merge multiple parsed Markdown table blocks from a single page into one
    unified list of rows.

    When OCR splits a logbook page at a stamp/annotation block mid-table, it
    produces two (or more) separate Markdown tables.  The second block usually
    lacks header rows (Month/Day columns), so normalize_logbook_rows drops all
    its data rows.  This function re-attaches those orphaned blocks to the
    first block's header so they are processed correctly.

    Strategy
    --------
    1. Find the first block that contains a proper header row (has Month & Day).
    2. Collect that header row (and any sub-header rows immediately after it).
    3. For every subsequent block: if it has its own header row, skip it (it's a
       duplicate); otherwise prepend the canonical header rows so that
       normalize_logbook_rows can locate Month/Day columns.
    4. Concatenate all data rows in page order.
    """
    if not all_block_rows:
        return []

    # Find the canonical header block index
    canonical_header_rows: list[list[str]] = []
    canonical_block_idx = -1
    for bi, block in enumerate(all_block_rows):
        for ri, row in enumerate(block):
            if _row_looks_like_header(row):
                canonical_block_idx = bi
                # Collect all header rows up to first data row
                for hri in range(ri, len(block)):
                    candidate = block[hri]
                    lc = [re.sub(r"\s+", " ", (c or "").strip()).lower() for c in candidate]
                    # stop when we hit a real data row (month abbreviation in month column)
                    month_idx_probe = lc.index("month") if "month" in lc else None
                    if month_idx_probe is not None:
                        canonical_header_rows.append(candidate)
                        continue
                    # sub-header rows: no month abbreviation in month col, no actual data
                    month_col = (
                        block[ri].index(
                            next(c for c in block[ri] if re.sub(r"\s+", " ", (c or "").strip()).lower() == "month")
                        )
                        if canonical_header_rows
                        else None
                    )
                    cell_in_month_col = (candidate[month_col].strip().upper() if month_col is not None and month_col < len(candidate) else "")
                    if cell_in_month_col in _MONTH_MAP or re.match(r"^\d+$", cell_in_month_col):
                        break  # hit first data row
                    canonical_header_rows.append(candidate)
                break
        if canonical_block_idx >= 0:
            break

    if not canonical_header_rows:
        # No proper header found — just concatenate all rows as-is
        merged: list[list[str]] = []
        for block in all_block_rows:
            merged.extend(block)
        return merged

    # Build merged row list
    merged = list(canonical_header_rows)  # start with canonical headers

    for bi, block in enumerate(all_block_rows):
        if bi == canonical_block_idx:
            # Add only the data rows from the canonical block (skip its own header rows)
            data_started = False
            header_row_count = len(canonical_header_rows)
            added = 0
            for row in block:
                lc = [re.sub(r"\s+", " ", (c or "").strip()).lower() for c in row]
                if not data_started:
                    if added < header_row_count:
                        added += 1
                        continue  # skip the header rows we already added
                    data_started = True
                merged.append(row)
        else:
            # Secondary block: skip any header-like rows, just take data rows
            for row in block:
                if _row_looks_like_header(row):
                    continue
                # Pad or trim to canonical column width
                ncols = len(canonical_header_rows[0]) if canonical_header_rows else len(row)
                if len(row) < ncols:
                    row = row + [""] * (ncols - len(row))
                elif len(row) > ncols:
                    row = row[:ncols]
                merged.append(row)

    return merged


def normalize_logbook_rows(
    rows: list[list[str]],
    *,
    ocr_text: str,
    date_format: str = "DD/MM/YYYY",
    time_format: str = "HHMM",
    duration_decimal_to_colon: bool = True,
) -> list[list[str]]:
    """
    Convert Month/Day(/Year) columns into a single Date column and normalize time columns.
    Handles multi-row headers (group headers spanning several rows) by flattening them.
    Expects first row to be headers.
    """
    if not rows:
        return rows
    rows = pad_rows_to_rectangular(rows)
    if len(rows) < 2:
        return rows

    num_cols = len(rows[0])

    def norm_cell(x: str) -> str:
        return re.sub(r"\s+", " ", (x or "").strip()).lower()

    # Find the main header row (contains Month and Day).
    header_row_idx = 0
    for i in range(min(8, len(rows))):
        lc = [norm_cell(c) for c in rows[i]]
        has_month_day = "month" in lc and "day" in lc
        if not has_month_day:
            continue
        header_row_idx = i
        break

    # Detect sub-header rows that follow the main header (before data starts).
    main_lc = [norm_cell(c) for c in rows[header_row_idx]]
    month_idx_probe = main_lc.index("month") if "month" in main_lc else None
    day_idx_probe = main_lc.index("day") if "day" in main_lc else None

    data_start = header_row_idx + 1
    for i in range(header_row_idx + 1, min(header_row_idx + 10, len(rows))):
        cells = [c.strip() for c in rows[i]]
        has_month = (
            month_idx_probe is not None
            and cells[month_idx_probe].upper() in _MONTH_MAP
        )
        has_day = (
            day_idx_probe is not None and re.match(r"^\d+$", cells[day_idx_probe])
        )
        row_text = " ".join(cells).lower()
        is_totals = "progressive" in row_text and "total" in row_text
        if has_month or has_day or is_totals:
            data_start = i
            break
        data_start = i + 1

    # Flatten multi-row headers into a single header row.
    if data_start > header_row_idx + 1:
        header_rows = [list(rows[j]) for j in range(header_row_idx, data_start)]

        row0_vals = [c.strip() for c in header_rows[0]]
        non_empty_pos = [j for j in range(num_cols) if row0_vals[j]]
        for k in range(len(non_empty_pos)):
            pos = non_empty_pos[k]
            end = non_empty_pos[k + 1] if k + 1 < len(non_empty_pos) else num_cols
            if end - pos <= 1:
                continue
            for hrow in header_rows:
                last = ""
                for j in range(pos, end):
                    val = hrow[j].strip() if j < len(hrow) else ""
                    if val:
                        last = val
                    elif last:
                        hrow[j] = last

        flattened: list[str] = []
        for col in range(num_cols):
            parts: list[str] = []
            seen: set[str] = set()
            for hrow in header_rows:
                val = hrow[col].strip() if col < len(hrow) else ""
                val_lc = val.lower()
                if val and val_lc not in seen:
                    parts.append(val)
                    seen.add(val_lc)
            combined = " ".join(parts)
            combined = re.sub(
                r"SINGLE[- ]ENGINE\s+FLIGHT\s+TIME\s*", "SE ", combined, flags=re.I
            )
            combined = re.sub(
                r"MULTI[- ]ENGINE\s+FLIGHT\s+TIME\s*", "ME ", combined, flags=re.I
            )
            combined = re.sub(r"\s+", " ", combined).strip()
            flattened.append(combined)

        headers = flattened
    else:
        headers = [str(h or "").strip() for h in rows[header_row_idx]]

    header_lc = [h.lower() for h in headers]

    month_idx = header_lc.index("month") if "month" in header_lc else None
    day_idx = header_lc.index("day") if "day" in header_lc else None
    year_idx = header_lc.index("year") if "year" in header_lc else None

    # Fallback: flattened headers can pick up annotations (e.g. "Month 1990"
    # when the year row was stacked onto the Month column). Match any header
    # that begins with the keyword as a standalone word.
    if month_idx is None:
        for i, h in enumerate(header_lc):
            if re.match(r"^month\b", h):
                month_idx = i
                break
    if day_idx is None:
        for i, h in enumerate(header_lc):
            if re.match(r"^day\b", h):
                day_idx = i
                break

    year_from_header: int | None = None
    if year_idx is None:
        for i, h in enumerate(header_lc):
            m_year = re.match(r"^year\s+(\d{4})", h)
            if m_year:
                year_idx = i
                year_from_header = int(m_year.group(1))
                break

    year_default = _detect_year_from_ocr_text(ocr_text)
    if year_default is None and year_from_header is not None:
        year_default = year_from_header

    # Grab a 4-digit year from any header cell (e.g. "Month 1990", "Year 1989-")
    # as a last-resort default so rows still get dated.
    if year_default is None:
        for h in header_lc:
            m_year = re.search(r"\b(19\d{2}|20\d{2})\b", h)
            if m_year:
                year_default = int(m_year.group(1))
                break

    if year_default is None and year_idx is not None:
        for r in rows[data_start:]:
            yr_raw = r[year_idx].strip() if year_idx < len(r) else ""
            if yr_raw:
                try:
                    yr_val = int(re.sub(r"\D", "", yr_raw) or "0")
                except ValueError:
                    continue
                if 1900 <= yr_val <= 2100:
                    year_default = yr_val
                    break

    time_header_names = {
        "out", "off", "on", "in", "departure", "arrival", "dep", "arr",
    }
    time_col_idxs = {i for i, h in enumerate(header_lc) if h in time_header_names}

    # Detect a single column whose flattened header contains BOTH "month" and "day"
    # as standalone words (logbooks where Month/Day are stacked sub-headers under
    # one column and data cells look like "JAN 27").
    combined_date_idx = None
    if month_idx is None and day_idx is None:
        for i, h in enumerate(header_lc):
            words = set(re.findall(r"\b(?:month|day)\b", h))
            if {"month", "day"}.issubset(words):
                combined_date_idx = i
                break

    if (
        month_idx is None
        and day_idx is None
        and year_idx is None
        and combined_date_idx is None
        and not time_col_idxs
    ):
        return rows

    drop_idxs = {
        i for i in (month_idx, day_idx, year_idx, combined_date_idx) if i is not None
    }
    new_headers: list[str] = []
    insert_date_at = None
    date_source_idxs = [
        i for i in (month_idx, day_idx, year_idx, combined_date_idx) if i is not None
    ]
    if date_source_idxs:
        insert_date_at = min(date_source_idxs)
    for i, h in enumerate(headers):
        if insert_date_at is not None and i == insert_date_at:
            new_headers.append("Date")
        if i in drop_idxs:
            continue
        new_headers.append(h)

    new_headers = [h.replace(",", " /") for h in new_headers]

    duration_col_idxs: set[int] = set()
    for idx, h in enumerate(new_headers):
        h_lc = h.lower()
        if h_lc == "date":
            continue
        if (
            re.search(r"\(\d+\)", h)
            or "totals" in h_lc
            or "flight time" in h_lc
        ):
            duration_col_idxs.add(idx)

    out_rows: list[list[str]] = [new_headers]

    current_year = year_default
    prev_month_seen: int | None = None
    seen_row_keys: set[tuple[str, ...]] = set()

    def _is_summary_row(cells: list[str]) -> bool:
        """Totals/summary rows that the logbook page prints below the flights."""
        joined = " ".join(c.strip() for c in cells if c.strip()).upper()
        if not joined:
            return True
        if "PROGRESSIVE TOTALS" in joined:
            return True
        if "GRAND TOTAL" in joined:
            return True
        if re.search(r"\bTOTAL\b\s*\d", joined):
            return True
        # Row with only numeric / dash / colon cells (a trailing per-column totals row).
        text_cells = [
            c.strip()
            for c in cells
            if c.strip() and not re.fullmatch(r"[\d\.\:\s\-]+", c.strip())
        ]
        return not text_cells

    for r in rows[data_start:]:
        r = ["" if v is None else str(v) for v in r]

        if not any(c.strip() for c in r):
            continue

        if _is_summary_row(r):
            continue

        row_key = tuple(c.strip() for c in r)
        if row_key in seen_row_keys:
            continue
        seen_row_keys.add(row_key)

        date_value = ""
        if insert_date_at is not None:
            y = None
            if year_idx is not None and year_from_header is None:
                yr_raw = r[year_idx].strip() if year_idx < len(r) else ""
                if yr_raw:
                    try:
                        y = int(re.sub(r"\D", "", yr_raw) or "0") or None
                    except ValueError:
                        y = None
            if y is None:
                y = year_default

            m = None
            if month_idx is not None:
                mon_raw = r[month_idx].strip() if month_idx < len(r) else ""
                mon_key = re.sub(r"[^A-Za-z]", "", mon_raw).upper()
                m = _MONTH_MAP.get(mon_key) or _MONTH_MAP.get(mon_key[:3])
                if m is None:
                    try:
                        m = int(re.sub(r"\D", "", mon_raw) or "0")
                    except ValueError:
                        m = None
            d = None
            if day_idx is not None:
                day_raw = r[day_idx].strip() if day_idx < len(r) else ""
                try:
                    d = int(re.sub(r"\D", "", day_raw) or "0")
                except ValueError:
                    d = None

            if combined_date_idx is not None:
                cell = (
                    r[combined_date_idx].strip()
                    if combined_date_idx < len(r)
                    else ""
                )
                pm, pd = _parse_combined_month_day(cell)
                if pm is not None:
                    m = pm
                if pd is not None:
                    d = pd

            # Year rollover: if the parsed month steps backwards relative to the
            # previous data row's month, the calendar year has advanced.
            if m is not None and (1 <= m <= 12):
                if (
                    prev_month_seen is not None
                    and current_year is not None
                    and m < prev_month_seen
                ):
                    current_year = current_year + 1
                prev_month_seen = m
                if year_idx is None or combined_date_idx is not None:
                    y = current_year

            if y and m and d and (1 <= m <= 12) and (1 <= d <= 31):
                date_value = _format_date(int(y), int(m), int(d), date_format)

        new_row: list[str] = []
        for i, val in enumerate(r):
            if insert_date_at is not None and i == insert_date_at:
                new_row.append(date_value)
            if i in drop_idxs:
                continue
            if i in time_col_idxs:
                new_row.append(_normalize_time_value(val, time_format))
            else:
                new_col_idx = len(new_row)
                if (
                    duration_decimal_to_colon
                    and new_col_idx in duration_col_idxs
                    and re.fullmatch(r"\d+\.\d+", val.strip())
                ):
                    new_row.append(val.strip().replace(".", ":", 1))
                else:
                    new_row.append(val)

        out_rows.append(new_row)

    return out_rows


def pad_rows_to_rectangular(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return rows
    max_cols = max(len(r) for r in rows)
    if max_cols <= 0:
        return rows
    return [r + [""] * (max_cols - len(r)) for r in rows]


def extract_all_table_block_lines(text: str) -> list[list[str]]:
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
    return [_parse_markdown_table_lines(block) for block in extract_all_table_block_lines(text)]


def strip_stamps(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, (str, bytes)):
        text = str(text)
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return re.sub(r"\[(STAMP|WATERMARK|ANNOTATION)\s*:[^\]]*\]", "", text, flags=re.IGNORECASE)


def _cell_looks_like_stamp_or_signature(cell: str) -> bool:
    if not cell or not str(cell).strip():
        return False
    raw = str(cell)
    t = " ".join(raw.split()).upper()
    tc = re.sub(r"\s+", "", t)

    if "[SIGNATURE]" in raw or "SIGNATURE]" in raw:
        return True

    stamp_phrases = (
        "FOGARTY", "AVIATIONACADEMY", "CORRECTTODATE", "CORRECT TO DATE",
        "ICONSIDER", "I CONSIDER", "CONSIDER", "COMPETENTTOFLY",
        "COMPETENT TO FLY", "SOLOBYDAY", "SOLO BY DAY", "SOLO BY NIGHT",
        "SOLO FLIGHT", "TYPEAIRCRAFT", "TYPE AIRCRAFT", "HE/SHEHASBEEN",
        "HE/SHE HAS BEEN", "INSTRUCTEDINANDFOUND", "ALLSEQUENCESFORTHIS",
        "ALL SEQUENCES FOR THIS", "COMPETENTINALL",
    )
    if any(p.replace(" ", "") in tc for p in stamp_phrases):
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


def _looks_like_valid_table_result(text: str) -> bool:
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
    last = ""
    for attempt in range(1, max(1, retries) + 1):
        try:
            text = ocr_image(image_path, verify=verify)
        except Exception as e:
            text = ""
            last = f""
        text = strip_stamps(text)
        last = text
        if _looks_like_valid_table_result(text):
            return text
    return last


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def _collect_image_paths(root: Path) -> list[Path]:
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
        "--retries",
        type=int,
        default=3,
        help="Retries per image when OCR output is empty/no-table (default: 3)",
    )
    parser.add_argument(
        "--date-format",
        default="DD/MM/YYYY",
        help="Date format for results.csv (DD/MM/YYYY, MM/DD/YYYY, or YYYY-MM-DD)",
    )
    parser.add_argument(
        "--time-format",
        default="HHMM",
        help='Time format for results.csv ("HHMM" or "HH:MM")',
    )
    parser.add_argument(
        "--duration-decimal-to-colon",
        action="store_true",
        help='Convert decimals like "249.5" to "249:5" in duration columns',
    )
    parser.add_argument(
        "--no-duration-decimal-to-colon",
        action="store_true",
        help='Disable converting decimals like "2.0" to "2:0" in duration columns',
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

    do_verify = not args.no_verify

    for image_path in images:
        target = str(image_path)

        result = ocr_with_retries(target, verify=do_verify, retries=args.retries)
        result = strip_stamps(result)

        print(result)

        # ------------------------------------------------------------------ #
        # KEY FIX: merge all table blocks from this page into one before      #
        # normalizing, so APRIL rows after the stamp block are not lost.      #
        # ------------------------------------------------------------------ #
        all_block_rows = extract_all_table_rows(result)

        # Clean stamp cells in every block first
        all_block_rows = [clean_stamp_cells_in_rows(b) for b in all_block_rows]
        all_block_rows = [pad_rows_to_rectangular(b) for b in all_block_rows]

        # Merge all blocks from this page into a single row list
        merged_rows = merge_table_blocks_for_page(all_block_rows)
        merged_rows = pad_rows_to_rectangular(merged_rows)

        # Save the raw OCR markdown (tables + headings) to results.csv.
        results_path = Path("results.csv")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(result or "")
            f.write("\n")

        # Normalize and write to csv.csv
        csv_path = Path("csv.csv")
        normalized = normalize_logbook_rows(
            merged_rows,
            ocr_text=result,
            date_format=args.date_format,
            time_format=args.time_format,
            duration_decimal_to_colon=not args.no_duration_decimal_to_colon,
        )
        table_csv = _csv_text_from_rows(normalized) if normalized else ""

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            if table_csv:
                f.write(table_csv)
                f.write("\n")

        log_path = Path("output_folder/csv_output/gem_runs.csv")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_exists = log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            if not log_exists:
                writer.writerow(["timestamp", "source", "output_file"])
            writer.writerow([datetime.now().isoformat(), target, str(csv_path)])