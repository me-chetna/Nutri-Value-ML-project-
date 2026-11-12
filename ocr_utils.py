"""OCR helpers for extracting nutrition facts from packaged-food labels."""
from functools import lru_cache
from io import BytesIO
import re

import numpy as np
from PIL import Image, ImageOps

FEATURES = ["calories", "sugar", "protein", "fat", "sodium", "carbohydrates", "fiber"]
NUMBER = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?")


@lru_cache(maxsize=1)
def _reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)


def extract_text(image_bytes: bytes) -> str:
    """Read text from an image supplied as bytes."""
    with Image.open(BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        # Upscale small label images to help OCR.
        if image.width < 1200:
            scale = 1200 / image.width
            image = image.resize((1200, int(image.height * scale)))
        array = np.asarray(image)
    lines = _reader().readtext(array, detail=0, paragraph=False)
    return "\n".join(str(line).strip() for line in lines if str(line).strip())


def _parse_number(text: str):
    match = NUMBER.search(text)
    if not match:
        return None, ""
    raw = match.group().replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None, ""
    tail = text[match.end():]
    unit_match = re.search(r"\b(kcal|kj|mg|g)\b", tail, re.I)
    unit = unit_match.group(1).lower() if unit_match else ""
    return value, unit


def extract_nutrition(text: str) -> dict:
    """Extract approximate values. Always ask the user to verify OCR results."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    patterns = {
        "calories": [r"\bcalories?\b", r"\benergy\b"],
        "sugar": [r"\btotal sugars?\b", r"(?<!added )\bsugars?\b"],
        "protein": [r"\bprotein\b"],
        "fat": [r"\btotal fat\b", r"(?<!saturated )(?<!trans )\bfat\b"],
        "sodium": [r"\bsodium\b", r"\bsalt\b"],
        "carbohydrates": [r"\btotal carbohydrates?\b", r"\bcarbohydrates?\b", r"\bcarbs?\b"],
        "fiber": [r"\bdietary fib(?:er|re)\b", r"\bfib(?:er|re)\b"],
    }
    result = {key: None for key in FEATURES}

    for key, label_patterns in patterns.items():
        found = False
        for i, line in enumerate(lines):
            for pattern in label_patterns:
                match = re.search(pattern, line, re.I)
                if not match:
                    continue
                # Avoid treating added sugar as total sugar or saturated fat as total fat.
                if key == "sugar" and re.search(r"\badded sugars?\b", line, re.I) and not re.search(r"\btotal sugars?\b", line, re.I):
                    continue
                if key == "fat" and re.search(r"\b(saturated|trans|unsaturated) fat\b", line, re.I) and not re.search(r"\btotal fat\b", line, re.I):
                    continue
                value, unit = _parse_number(line[match.end():])
                if value is None and i + 1 < len(lines):
                    value, unit = _parse_number(lines[i + 1])
                if value is None:
                    continue
                if key == "calories" and unit == "kj":
                    value /= 4.184
                elif key == "sodium":
                    if unit == "g":
                        value *= 1000
                    elif re.search(r"\bsalt\b", match.group(), re.I):
                        # Salt is approximately 2.5 times sodium by mass.
                        value = value * 400 if unit in ("", "g") else value * 0.4
                elif key in {"sugar", "protein", "fat", "carbohydrates", "fiber"} and unit == "mg":
                    value /= 1000
                result[key] = float(value)
                found = True
                break
            if found:
                break
    return result
