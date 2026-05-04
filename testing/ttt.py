from pathlib import Path
import csv

in_path = Path(r"test_file\page_1_gemini_new_prompt_update.txt")
out_path = in_path.with_suffix(".csv")

lines = in_path.read_text(encoding="utf-8", errors="replace").splitlines()

# keep only table-like lines (contain |)
table_lines = [ln for ln in lines if "|" in ln]

rows = []
for ln in table_lines:
    s = ln.strip()
    # handle lines like: | a | b | c |
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    cells = [c.strip() for c in s.split("|")]
    rows.append(cells)

# make all rows same length
max_cols = max((len(r) for r in rows), default=0)
rows = [r + [""] * (max_cols - len(r)) for r in rows]

with out_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerows(rows)

print(f"Wrote: {out_path}")