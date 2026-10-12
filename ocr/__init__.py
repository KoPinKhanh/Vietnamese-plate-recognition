import os
from pathlib import Path

# Auto-configure Tesseract path if present on Windows
TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if TESSERACT_EXE.exists():
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)
    except ImportError:
        pass

from .plate_text_utils import normalize_plate_text, calculate_cer, is_exact_match, format_plate_text
from .preprocess import preprocess_plate, PREPROCESS_STRATEGIES

__all__ = [
    "normalize_plate_text",
    "calculate_cer",
    "is_exact_match",
    "format_plate_text",
    "preprocess_plate",
    "PREPROCESS_STRATEGIES",
]

