#!/usr/bin/env python3
"""Check whether Gemini and Google Vision APIs are reachable and valid."""

from __future__ import annotations

import argparse
import base64
import os
import sys

from dotenv import load_dotenv

load_dotenv()

MINIMAL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAAl21bKAAAAA1BMVEX///+nxBvIAAAAAXRSTlMAQObYZgAAAApJREFUCNdjYAAAAAIAAeIhvDMAAAAASUVORK5CYII="
)

def check_google_vision() -> tuple[bool, str]:
    try:
        from google.cloud import vision
    except Exception as exc:  # pragma: no cover
        return False, f"Import failed: {exc}"

    api_key = os.getenv("GOOGLE_VISION_API_KEY")
    if api_key:
        client = vision.ImageAnnotatorClient(client_options={"api_key": api_key})
    else:
        client = vision.ImageAnnotatorClient()

    image_bytes = base64.b64decode(MINIMAL_PNG_BASE64)
    image = vision.Image(content=image_bytes)

    try:
        response = client.label_detection(image=image)
    except Exception as exc:
        return False, f"API request failed: {exc}"

    if getattr(response, "error", None) and getattr(response.error, "message", None):
        return False, response.error.message

    labels = [label.description for label in getattr(response, "label_annotations", [])][:5]
    if labels:
        return True, f"OK (labels: {', '.join(labels)})"

    return True, "OK (no labels returned)"


def check_gemini() -> tuple[bool, str]:
    try:
        from google.genai import Client
    except Exception as exc:  # pragma: no cover
        return False, f"Import failed: {exc}"

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False, "Missing environment variable GOOGLE_API_KEY"

    client = Client(api_key=api_key)
    candidate_models = [
        "models/gemini-2.5-flash",
        "models/gemini-2.1",
        "models/text-bison-001",

        # "models/gemini-2.5-flash",
        # "models/gemini-2.0-flash",
        # "models/gemini-flash-latest",
        # "models/gemini-pro-latest"
    ]

    errors: list[str] = []
    for model in candidate_models:
        try:
            response = client.models.generate_content(
                model=model,
                contents="Ping",
            )
            text = getattr(response, "text", None)
            if text:
                return True, f"OK with model {model}: {text.strip()[:120]!r}"
            return True, f"OK with model {model}: response received"
        except Exception as exc:
            errors.append(f"{model}: {exc}")

    return False, "All candidate models failed: " + " | ".join(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Check Gemini / Google GenAI connectivity.",
    )
    parser.add_argument(
        "--google-vision",
        action="store_true",
        help="Check Google Vision connectivity.",
    )
    args = parser.parse_args()

    if not args.gemini and not args.google_vision:
        args.gemini = True
        args.google_vision = True

    success = True

    if args.google_vision:
        ok, message = check_google_vision()
        print("Google Vision:", "✅" if ok else "❌", message)
        success = success and ok

    if args.gemini:
        ok, message = check_gemini()
        print("Gemini:", "✅" if ok else "❌", message)
        success = success and ok

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
