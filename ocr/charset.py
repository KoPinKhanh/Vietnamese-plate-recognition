"""
Character set definitions and encoding/decoding utilities for Vietnamese License Plate OCR.
Standard vocabulary: 10 digits (0-9) + 26 uppercase English letters (A-Z) = 36 characters.
CTC blank index is placed at index 36, giving 37 total classes.
"""

from typing import List, Dict, Optional
from ocr.plate_text_utils import normalize_plate_text

# 36 alphanumeric characters
CHARSET: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CHARS: int = len(CHARSET)  # 36
BLANK_INDEX: int = NUM_CHARS   # 36 (0-indexed, so 37th class)
NUM_CLASSES: int = NUM_CHARS + 1  # 37

# Lookup dictionaries
CHAR_TO_IDX: Dict[str, int] = {char: idx for idx, char in enumerate(CHARSET)}
IDX_TO_CHAR: Dict[int, str] = {idx: char for idx, char in enumerate(CHARSET)}
IDX_TO_CHAR[BLANK_INDEX] = "<blank>"


def get_charset() -> str:
    """Returns the standard charset string."""
    return CHARSET


def get_num_classes() -> int:
    """Returns total output classes including CTC blank."""
    return NUM_CLASSES


def get_blank_index() -> int:
    """Returns the index reserved for the CTC blank symbol."""
    return BLANK_INDEX


def encode_text(text: str) -> List[int]:
    """
    Normalizes input text and encodes each character into its vocabulary index.
    Ignores characters outside the vocabulary.
    
    Args:
        text: Raw plate string (e.g. "30A-123.45").
        
    Returns:
        List of integer indices (e.g. [3, 0, 10, 1, 2, 3, 4, 5]).
    """
    clean_text = normalize_plate_text(text)
    encoded = []
    for c in clean_text:
        if c in CHAR_TO_IDX:
            encoded.append(CHAR_TO_IDX[c])
    return encoded


def decode_indices(indices: List[int]) -> str:
    """
    Converts list of valid character indices (excluding blank) to a string.
    
    Args:
        indices: List of integer indices.
        
    Returns:
        Decoded string.
    """
    chars = []
    for idx in indices:
        if idx in IDX_TO_CHAR and idx != BLANK_INDEX:
            chars.append(IDX_TO_CHAR[idx])
    return "".join(chars)
