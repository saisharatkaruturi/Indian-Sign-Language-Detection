"""Finger-count numbers (0-5) on synthetic MediaPipe-style landmarks (no camera, no model)."""

import unittest
from types import SimpleNamespace as P

from src.inference.gesture_translator import ISLGestureTranslator
from src.inference.live_test import NUMBER_TARGETS, target_key


def hand(raised_fingers, thumb_out):
    """21 landmarks of an upright right hand; wrist at the bottom, fingers pointing up."""
    lm = [P(x=0.5, y=0.9)] * 21
    lm = list(lm)
    lm[0] = P(x=0.5, y=0.9)                                   # wrist
    columns = {"index": 0.42, "middle": 0.5, "ring": 0.58, "pinky": 0.66}
    ids = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
    for name, (mcp, pip, dip, tip) in ids.items():
        x = columns[name]
        lm[mcp], lm[pip] = P(x=x, y=0.62), P(x=x, y=0.52)
        if name in raised_fingers:
            lm[dip], lm[tip] = P(x=x, y=0.45), P(x=x, y=0.38)
        else:  # curled back toward the palm
            lm[dip], lm[tip] = P(x=x, y=0.58), P(x=x, y=0.66)
    lm[1], lm[2], lm[3] = P(x=0.42, y=0.82), P(x=0.36, y=0.75), P(x=0.32, y=0.70)
    lm[4] = P(x=0.24, y=0.64) if thumb_out else P(x=0.45, y=0.66)
    return lm


class TestFingerCount(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.t = ISLGestureTranslator.__new__(ISLGestureTranslator)  # rule methods only, no MediaPipe/CNN

    def test_counts_zero_to_five(self):
        cases = {
            0: ([], False),
            1: (["index"], False),
            2: (["index", "middle"], False),
            3: (["index", "middle", "ring"], False),
            4: (["index", "middle", "ring", "pinky"], False),
            5: (["index", "middle", "ring", "pinky"], True),
        }
        for expected, (fingers, thumb) in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(self.t.count_raised_fingers(hand(fingers, thumb)), expected)

    def test_scale_invariant(self):
        far = [P(x=0.5 + (p.x - 0.5) * 0.4, y=0.5 + (p.y - 0.5) * 0.4) for p in hand(["index", "middle"], True)]
        self.assertEqual(self.t.count_raised_fingers(far), 3)


class TestTargets(unittest.TestCase):

    def test_number_labels_and_targets(self):
        self.assertEqual(NUMBER_TARGETS, ["0", "1", "2", "3", "4", "5"])
        self.assertEqual(target_key("Number '3'"), "3")
        self.assertEqual(target_key("Sign 'A' (Fist)"), "A")
        self.assertEqual(target_key("Use ONE hand for numbers"), "Use ONE hand for numbers")


if __name__ == "__main__":
    unittest.main()
