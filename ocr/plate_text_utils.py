"""
Utilities for cleaning, normalizing, and evaluating license plate text.
"""

import re
from typing import Tuple


def normalize_plate_text(text: str) -> str:
    """
    Normalizes OCR result to standard Vietnamese license plate character string:
    - Uppercase
    - Strips spaces, newlines, dots, hyphens, and any non-alphanumeric characters
    - Keeps only characters [A-Z0-9]
    """
    if not text:
        return ""
    
    cleaned = text.strip().upper()
    # Remove any non-alphanumeric character
    cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)
    return cleaned


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def calculate_cer(ground_truth: str, predicted: str) -> float:
    """
    Calculates Character Error Rate (CER) = LevenshteinDistance / len(ground_truth).
    Returns 0.0 for empty ground truth when predicted is also empty, else 1.0.
    """
    gt_norm = normalize_plate_text(ground_truth)
    pred_norm = normalize_plate_text(predicted)
    
    if len(gt_norm) == 0:
        return 0.0 if len(pred_norm) == 0 else 1.0
    
    dist = levenshtein_distance(gt_norm, pred_norm)
    return dist / len(gt_norm)


def is_exact_match(ground_truth: str, predicted: str) -> bool:
    """Returns True if normalized ground truth matches normalized prediction exactly."""
    return normalize_plate_text(ground_truth) == normalize_plate_text(predicted)


def format_plate_text(text: str, layout: str = "auto") -> str:
    """
    Formats normalized plate text for readable presentation.
    E.g.:
    - '30A12345' -> '30A - 123.45' (standard 5-digit)
    - '30A1234' -> '30A - 1234' (standard 4-digit)
    - '29M112345' -> '29-M1 123.45'
    """
    s = normalize_plate_text(text)
    if len(s) == 8:
        # e.g., 30A12345 -> 30A-123.45
        return f"{s[:3]}-{s[3:6]}.{s[6:]}"
    elif len(s) == 7:
        # e.g., 30A1234 -> 30A-1234
        return f"{s[:3]}-{s[3:]}"
    elif len(s) == 9:
        # e.g., 29M112345 -> 29M1-123.45 or 30LD12345
        return f"{s[:4]}-{s[4:7]}.{s[7:]}"
    return s
