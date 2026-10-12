"""
Unit tests for CTC greedy decoder.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import unittest
from ocr.charset import CHAR_TO_IDX, BLANK_INDEX, NUM_CLASSES
from ocr.ctc_decoder import (
    ctc_greedy_decode_indices,
    ctc_greedy_decode_numpy,
    ctc_greedy_decode_torch,
)


class TestCTCDecoder(unittest.TestCase):
    def test_collapse_and_blank_removal(self):
        # Sample sequence: [B, 4, 4, B, 3, 3, A, B, 1, 1] -> "43A1"
        B = BLANK_INDEX
        idx_4 = CHAR_TO_IDX["4"]
        idx_3 = CHAR_TO_IDX["3"]
        idx_A = CHAR_TO_IDX["A"]
        idx_1 = CHAR_TO_IDX["1"]

        seq = [B, idx_4, idx_4, B, idx_3, idx_3, idx_A, B, idx_1, idx_1]
        decoded = ctc_greedy_decode_indices(seq, blank_index=B)
        self.assertEqual(decoded, [idx_4, idx_3, idx_A, idx_1])

    def test_consecutive_identical_with_and_without_blank(self):
        B = BLANK_INDEX
        idx_8 = CHAR_TO_IDX["8"]
        # Without blank: 8, 8, 8 -> '8'
        self.assertEqual(ctc_greedy_decode_indices([idx_8, idx_8, idx_8], B), [idx_8])
        # With blank: 8, B, 8 -> '88'
        self.assertEqual(ctc_greedy_decode_indices([idx_8, B, idx_8], B), [idx_8, idx_8])
        # Three 8s: 8, 8, B, 8, B, 8, 8 -> '888'
        self.assertEqual(ctc_greedy_decode_indices([idx_8, idx_8, B, idx_8, B, idx_8, idx_8], B), [idx_8, idx_8, idx_8])

    def test_all_blanks_and_empty(self):
        B = BLANK_INDEX
        self.assertEqual(ctc_greedy_decode_indices([B, B, B], B), [])
        self.assertEqual(ctc_greedy_decode_indices([], B), [])

    def test_numpy_decoding(self):
        B = BLANK_INDEX
        T = 20
        # Create logits [37, 20]
        logits = np.zeros((NUM_CLASSES, T), dtype=np.float32)
        # Default all timesteps to blank
        logits[B, :] = 10.0

        # Inject '4' at t=2,3; '3' at t=5,6; 'A' at t=8; '1' at t=11,12
        chars = [('4', [2, 3]), ('3', [5, 6]), ('A', [8]), ('1', [11, 12])]
        for c, timesteps in chars:
            for t in timesteps:
                logits[CHAR_TO_IDX[c], t] = 20.0

        pred = ctc_greedy_decode_numpy(logits, blank_index=B)
        self.assertEqual(pred, "43A1")

    def test_torch_batch_decoding(self):
        B = BLANK_INDEX
        batch_size = 2
        T = 20
        logits = torch.zeros(batch_size, NUM_CLASSES, T, dtype=torch.float32)
        logits[:, B, :] = 10.0

        # Sample 1: "29A1"
        for t in [1, 2]: logits[0, CHAR_TO_IDX['2'], t] = 20.0
        for t in [4, 5]: logits[0, CHAR_TO_IDX['9'], t] = 20.0
        for t in [7, 8]: logits[0, CHAR_TO_IDX['A'], t] = 20.0
        for t in [10, 11]: logits[0, CHAR_TO_IDX['1'], t] = 20.0

        # Sample 2: "51F8888"
        for t in [1]: logits[1, CHAR_TO_IDX['5'], t] = 20.0
        for t in [3]: logits[1, CHAR_TO_IDX['1'], t] = 20.0
        for t in [5]: logits[1, CHAR_TO_IDX['F'], t] = 20.0
        for t in [7]: logits[1, CHAR_TO_IDX['8'], t] = 20.0
        # blank at t=8
        for t in [9]: logits[1, CHAR_TO_IDX['8'], t] = 20.0
        # blank at t=10
        for t in [11]: logits[1, CHAR_TO_IDX['8'], t] = 20.0
        # blank at t=12
        for t in [13]: logits[1, CHAR_TO_IDX['8'], t] = 20.0

        preds = ctc_greedy_decode_torch(logits, blank_index=B)
        self.assertEqual(preds, ["29A1", "51F8888"])


if __name__ == "__main__":
    unittest.main()
