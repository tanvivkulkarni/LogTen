def convert_pdf_to_rotated_pngs(pdf_path: str, output: str = "output", dpi: int = 300) -> list[str]:
    from pdf2image import convert_from_path
    import os
    from PIL import Image
    from PIL import ImageEnhance
    import pytesseract
    import cv2
    import numpy as np

    os.makedirs(output, exist_ok=True)
    original_dir = os.path.join(output, "original")
    enhanced_dir = os.path.join(output, "enhanced")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)

    pages = convert_from_path(pdf_path, dpi=dpi)

    saved_paths: list[str] = []

    for i, page in enumerate(pages, start=1):
        # Convert PIL image to OpenCV format
        cv_img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

        # Use pytesseract OSD to detect orientation
        osd = pytesseract.image_to_osd(page)
        rotate_angle = int([line for line in osd.split("\n") if "Rotate:" in line][0].split(":")[1].strip())

        # Rotate image if needed
        if rotate_angle != 0:
            rotate_dict = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE
            }
            cv_img = cv2.rotate(cv_img, rotate_dict[rotate_angle])

        # Save rotated "original" image
        original_path = os.path.join(original_dir, f"page_{i}.png")
        cv2.imwrite(original_path, cv_img)
        print(f"Saved correctly rotated: {original_path}")

        # Enhance image (contrast + sharpness) and save in `enhanced/`
        pil_rotated = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        enhanced = ImageEnhance.Contrast(pil_rotated).enhance(3.0)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(3.0)
        enhanced_path = os.path.join(enhanced_dir, f"page_{i}.png")
        enhanced.save(enhanced_path)
        print(f"Saved enhanced: {enhanced_path}")

        # Use enhanced images downstream
        saved_paths.append(enhanced_path)

    print("All pages converted and correctly rotated successfully.")
    return saved_paths


def _get_level(index, hierarchy) -> int:
    level = 0
    parent = hierarchy[index][3]
    while parent != -1:
        level += 1
        parent = hierarchy[parent][3]
    return level


def dataframe_from_page_boxes(image_path: str):
    import cv2
    import numpy as np
    import pandas as pd
    import os
    from PIL import Image
    from doctr.models import ocr_predictor
    from doctr.io import DocumentFile

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    output_with_rectangles = img.copy()
    row_column_canvas = np.ones_like(img) * 255

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert light background to white
    _, white_bg = cv2.threshold(gray, 223, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(white_bg)

    # Save filtered images in `filtered/` folder under the output root
    # Example: <output>/enhanced/page_1.png -> <output>/filtered/page_1_*.png
    parent_dir = os.path.dirname(image_path)              # .../<output>/enhanced
    output_root = os.path.dirname(parent_dir)             # .../<output>
    filtered_dir = os.path.join(output_root, "filtered")  # .../<output>/filtered
    os.makedirs(filtered_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(filtered_dir, f"{base}_white.png"), white_bg)
    cv2.imwrite(os.path.join(filtered_dir, f"{base}_white_inverted.png"), inverted)

    contours, hierarchy = cv2.findContours(inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(hierarchy) == 0:
        df_empty = pd.DataFrame(
            columns=["row", "column", "label", "x", "y", "x2", "y2", "words", "text"]
        )
        return df_empty

    hierarchy = hierarchy[0]

    # collect boxes first
    boxes = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        level = _get_level(i, hierarchy)

        if level > 3:
            continue

        if level == 0:
            color = (0, 255, 0)
        elif level == 1:
            color = (255, 0, 0)
        else:
            continue

        if w < 30 or h < 15:
            continue

        boxes.append((x, y, w, h, color))

    # sort row-wise
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    row_tolerance = 20
    current_row = 1
    current_col = 1
    prev_y = None
    box_records = []

    for x, y, w, h, color in boxes:
        # detect new row
        if prev_y is not None and abs(y - prev_y) > row_tolerance:
            current_row += 1
            current_col = 1

        label = f"R{current_row}C{current_col}"
        x2 = x + w
        y2 = y + h

        # draw on original and on blank canvas (like `doctr_detection.py`)
        cv2.rectangle(output_with_rectangles, (x, y), (x2, y2), color, 2)
        cv2.rectangle(row_column_canvas, (x, y), (x2, y2), color, 2)
        cv2.putText(
            row_column_canvas,
            label,
            (x + 5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        box_records.append(
            {
                "row": current_row,
                "column": current_col,
                "label": label,
                "x": x,
                "y": y,
                "x2": x2,
                "y2": y2,
                "words": [],
            }
        )

        current_col += 1
        prev_y = y

    # Save the "drawn boxes" images under the output root
    boxes_dir = os.path.join(output_root, "boxes")
    os.makedirs(boxes_dir, exist_ok=True)
    rect_path = os.path.join(boxes_dir, f"{base}_with_rectangles.png")
    canvas_path = os.path.join(boxes_dir, f"{base}_row_column_canvas.png")
    cv2.imwrite(rect_path, output_with_rectangles)
    cv2.imwrite(canvas_path, row_column_canvas)

    # OCR
    model = ocr_predictor(pretrained=True, detect_orientation=True)
    doc = DocumentFile.from_images(image_path)
    result = model(doc)

    # Get original image size
    pil_img = Image.open(image_path)
    img_width, img_height = pil_img.size

    # PDF output like `doctr_detection.py`: use the row/column canvas as background and draw OCR text
    try:
        from reportlab.pdfgen import canvas as pdf_canvas

        pdf_dir = os.path.join(output_root, "pdf")
        os.makedirs(pdf_dir, exist_ok=True)
        pdf_file = os.path.join(pdf_dir, f"{base}_output_image_canvas.pdf")
        c = pdf_canvas.Canvas(pdf_file, pagesize=(img_width, img_height))
        c.drawImage(
            canvas_path,
            x=0,
            y=0,
            width=img_width,
            height=img_height,
            preserveAspectRatio=True,
            mask="auto",
        )
        c.setFillColorRGB(1, 0, 0)

        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        text = word.value
                        (x1, y1), (x2n, y2n) = word.geometry
                        x = round(x1 * img_width, 2)
                        y = round(img_height - y2n * img_height, 2)
                        c.drawString(x, y, text)

        c.save()
        print(f"PDF with image canvas saved → {pdf_file}")
    except ModuleNotFoundError as e:
        print("Skipping PDF export (missing dependency):", e)

    # Map OCR words into detected rectangles (box_records)
    # Doctr word.geometry is normalized with origin at top-left of the image.
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    text = word.value
                    (wx1n, wy1n), (wx2n, wy2n) = word.geometry
                    wx1 = wx1n * img_width
                    wy1 = wy1n * img_height
                    wx2 = wx2n * img_width
                    wy2 = wy2n * img_height

                    cx = (wx1 + wx2) / 2.0
                    cy = (wy1 + wy2) / 2.0

                    # choose the smallest-area rectangle that contains the word center
                    best_idx = None
                    best_area = None
                    for i, rec in enumerate(box_records):
                        if rec["x"] <= cx <= rec["x2"] and rec["y"] <= cy <= rec["y2"]:
                            area = (rec["x2"] - rec["x"]) * (rec["y2"] - rec["y"])
                            if best_area is None or area < best_area:
                                best_area = area
                                best_idx = i

                    if best_idx is not None:
                        box_records[best_idx]["words"].append(text)

    df_boxes = pd.DataFrame(box_records)
    df_boxes["text"] = df_boxes["words"].apply(lambda ws: " ".join(ws))
    return df_boxes


def save_page_dfs(image_paths: list[str], output: str) -> list[str]:
    import os

    df_dir = os.path.join(output, "df")
    os.makedirs(df_dir, exist_ok=True)

    saved_df_paths: list[str] = []
    for image_path in image_paths:
        df = dataframe_from_page_boxes(image_path)
        base = os.path.splitext(os.path.basename(image_path))[0]
        df_path = os.path.join(df_dir, f"{base}.tsv")
        df.drop(columns=["words"], errors="ignore").to_csv(df_path, index=False, sep="\t")
        print(f"Saved df: {df_path}")
        saved_df_paths.append(df_path)

    return saved_df_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PDF pages to images with correct orientation")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", default="output", help="Output folder name")

    args = parser.parse_args()
    saved_paths = convert_pdf_to_rotated_pngs(args.pdf_path, output=args.output)
    print(saved_paths)