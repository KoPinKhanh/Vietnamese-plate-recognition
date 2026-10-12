"""
CTC Greedy Decoder for Vietnamese License Plate OCR.
Supports both PyTorch tensors and NumPy arrays (for pure OpenCV DNN deployment).
"""

from typing import List, Union
import numpy as np
import torch
from ocr.charset import BLANK_INDEX, IDX_TO_CHAR, decode_indices


def ctc_greedy_decode_indices(indices: List[int], blank_index: int = BLANK_INDEX) -> List[int]:
    """
    Performs standard CTC greedy collapse on a 1D sequence of class indices:
    1. Collapse adjacent repeated symbols: [A, A, B, B, B] -> [A, B]
    2. Filter out CTC blank tokens: [A, BLANK, B] -> [A, B]
    
    Args:
        indices: 1D list of class predictions along time dimension.
        blank_index: Reserved index for blank token.
        
    Returns:
        List of decoded valid character indices.
    """
    collapsed = []
    prev_idx = None
    for idx in indices:
        if idx != prev_idx:
            if idx != blank_index:
                collapsed.append(idx)
            prev_idx = idx
    return collapsed


def ctc_greedy_decode_numpy(logits: np.ndarray, blank_index: int = BLANK_INDEX) -> str:
    """
    Decodes single-sample logits/probabilities of shape [C, T] or [T, C] or [1, C, T] using pure NumPy.
    
    Args:
        logits: NumPy array containing raw logits or probabilities.
        blank_index: Index of blank class.
        
    Returns:
        Decoded license plate text.
    """
    arr = np.squeeze(logits)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array [C, T] or [T, C], got shape {logits.shape}")

    # Determine time axis vs class axis
    # If shape is [37, T], argmax over axis 0 gives array of shape [T]
    # If shape is [T, 37], argmax over axis 1 gives array of shape [T]
    if arr.shape[0] == (blank_index + 1) or arr.shape[0] < arr.shape[1]:
        # [C, T] -> argmax along axis 0
        pred_indices = np.argmax(arr, axis=0).tolist()
    else:
        # [T, C] -> argmax along axis 1
        pred_indices = np.argmax(arr, axis=1).tolist()

    collapsed = ctc_greedy_decode_indices(pred_indices, blank_index=blank_index)
    return decode_indices(collapsed)


def ctc_greedy_decode_torch(logits: torch.Tensor, blank_index: int = BLANK_INDEX) -> List[str]:
    """
    Batch greedy decode for PyTorch tensor of shape [B, C, T] or [T, B, C].
    
    Args:
        logits: PyTorch tensor.
        blank_index: Index of blank class.
        
    Returns:
        List of decoded string predictions, length B.
    """
    if logits.ndim == 3 and logits.shape[1] == (blank_index + 1):
        # [B, C, T] -> argmax along C (dim=1) -> [B, T]
        pred_indices = torch.argmax(logits, dim=1).detach().cpu().numpy()
    elif logits.ndim == 3 and logits.shape[2] == (blank_index + 1):
        # [T, B, C] -> permute to [B, T, C] -> argmax along C (dim=2) -> [B, T]
        pred_indices = torch.argmax(logits.permute(1, 0, 2), dim=2).detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported logits shape {logits.shape} for blank_index {blank_index}")

    results = []
    for b in range(pred_indices.shape[0]):
        row = pred_indices[b].tolist()
        collapsed = ctc_greedy_decode_indices(row, blank_index=blank_index)
        results.append(decode_indices(collapsed))
    return results
