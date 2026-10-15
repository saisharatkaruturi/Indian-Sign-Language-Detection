"""Sign-triggered app launch tests. A FAKE opener records URLs instead of opening a browser."""

import unittest

from src.inference.actions import SignActionTrigger, build_trigger_from_config, sign_key
from src.utils.config import load_config


class TestSignKey(unittest.TestCase):

    def test_labels_from_live_pipeline(self):
        self.assertEqual(sign_key("Sign 'W' (WhatsApp)"), "W")
        self.assertEqual(sign_key("Sign 'Y' (Call / Phone)"), "Y")
        self.assertEqual(sign_key("Sign 'L'"), "L")
        self.assertIsNone(sign_key("Thumbs Up"))
        self.assertIsNone(sign_key("Keep palm & fingers in view"))
        self.assertIsNone(sign_key(""))


class TestTrigger(unittest.TestCase):

    def setUp(self):
        self.opened = []
        self.t = SignActionTrigger(hold_seconds=1.0, cooldown_seconds=5.0, min_confidence=0.7,
                                   opener=lambda url: self.opened.append(url) or True)

    def test_hold_opens_mapped_app_once(self):
        _, p, fired = self.t.update("Sign 'Y' (Call / Phone)", 0.96, 0.0)
        self.assertIsNone(fired)
        self.assertEqual(p, 0.0)
        held, p, fired = self.t.update("Sign 'Y' (Call / Phone)", 0.96, 0.5)
        self.assertEqual(held.name, "YouTube")
        self.assertAlmostEqual(p, 0.5)
        _, _, fired = self.t.update("Sign 'Y' (Call / Phone)", 0.96, 1.0)
        self.assertEqual(fired.name, "YouTube")
        self.assertEqual(self.opened, ["https://www.youtube.com"])
        for k in range(20):  # keep holding inside the cooldown: no second launch
            self.t.update("Sign 'Y' (Call / Phone)", 0.96, 1.1 + k * 0.2)
        self.assertEqual(len(self.opened), 1)

    def test_each_default_mapping(self):
        now = 0.0
        for label, url in (("Sign 'W' (WhatsApp)", "https://web.whatsapp.com"),
                           ("Sign 'L'", "https://www.google.com")):
            self.t.update(label, 0.95, now)
            self.t.update(label, 0.95, now + 1.0)
            self.assertEqual(self.opened[-1], url)
            now += 10.0

    def test_low_confidence_or_unmapped_sign_never_fires(self):
        for i in range(30):
            self.t.update("Sign 'W' (WhatsApp)", 0.5, i * 0.1)
            self.t.update("Thumbs Up", 0.98, i * 0.1)
        self.assertEqual(self.opened, [])

    def test_switching_sign_restarts_hold(self):
        self.t.update("Sign 'W' (WhatsApp)", 0.95, 0.0)
        self.t.update("Sign 'L'", 0.95, 0.9)
        _, p, fired = self.t.update("Sign 'L'", 0.95, 1.2)
        self.assertIsNone(fired)
        self.assertAlmostEqual(p, 0.3)

    def test_unsafe_url_rejected(self):
        with self.assertRaises(ValueError):
            SignActionTrigger(signs={"X": {"name": "Calc", "url": "C:/Windows/System32/calc.exe"}})


class TestConfig(unittest.TestCase):

    def test_project_config_maps_whatsapp_youtube_google(self):
        trig = build_trigger_from_config(load_config())
        self.assertEqual({k: a.name for k, a in trig.actions.items()}, {"W": "WhatsApp", "Y": "YouTube", "L": "Google"})

    def test_disabled(self):
        self.assertIsNone(build_trigger_from_config({"actions": {"enabled": False}}))


if __name__ == "__main__":
    unittest.main()
