"""Guided live letter test bookkeeping (no camera)."""

import tempfile
import unittest

import pandas as pd

from src.inference.live_test import SignTestSession


class TestSignTestSession(unittest.TestCase):

    def test_majority_vote_pass_fail_skip_and_summary(self):
        s = SignTestSession(["A", "B", "C"], capture_seconds=1.0)
        s.start_capture(0.0)
        for i, lbl in enumerate(["Sign 'A' (Fist)", "Sign 'A' (Fist)", "Sign 'S'", "Sign 'A' (Fist)"]):
            res = s.feed(lbl, i * 0.4)
        self.assertEqual(res["target"], "A")
        self.assertEqual(res["recognised"], "A")
        self.assertTrue(res["correct"])
        self.assertEqual(s.current, "B")
        s.start_capture(10.0)
        s.feed("Sign 'V' (Peace / Two)", 10.0)
        res = s.feed("Sign 'V' (Peace / Two)", 11.0)
        self.assertFalse(res["correct"])
        self.assertEqual(s.skip()["status"], "skipped")
        self.assertTrue(s.done)
        self.assertEqual(s.summary(), {"targets": 3, "tested": 2, "skipped": 1, "correct": 1,
                                       "accuracy_on_tested": 0.5, "failed": ["B->V"]})

    def test_feed_ignored_until_capture_started(self):
        s = SignTestSession(["A"])
        self.assertIsNone(s.feed("Sign 'A'", 0.0))
        self.assertEqual(s.results, [])

    def test_save_writes_csv(self):
        s = SignTestSession(["A"], capture_seconds=0.0)
        s.start_capture(0.0)
        s.feed("Sign 'A'", 0.0)
        with tempfile.TemporaryDirectory() as d:
            df = pd.read_csv(s.save(d))
        self.assertEqual(df.loc[0, "target"], "A")
        self.assertTrue(bool(df.loc[0, "correct"]))


if __name__ == "__main__":
    unittest.main()
