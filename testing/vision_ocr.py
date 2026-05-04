import csv
import os
from datetime import datetime

from dotenv import load_dotenv
from google.cloud import vision

load_dotenv()


def detect_document(path):
    """Detects document features in an image and returns the extracted text."""
    api_key = os.getenv("GOOGLE_VISION_API_KEY")
    if api_key:
        client = vision.ImageAnnotatorClient(client_options={"api_key": api_key})
    else:
        client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    return response.full_text_annotation.text if response.full_text_annotation.text else ""


def process_image(image_path, output_csv):
    """Process a single image and save OCR result to a CSV."""
    print(f"Processing: {image_path}")
    text = detect_document(image_path)
    timestamp = datetime.now().isoformat()

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "source", "ocr_text"])
        writer.writerow([timestamp, image_path, text])

    print(f"Done — extracted {len(text)} chars")
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR using Google Vision API")
    parser.add_argument("path", help="Path to the image file")
    args = parser.parse_args()

    process_image(args.path, "vision_output.csv")