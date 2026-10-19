"""Pre-rename environment names still work for one release, behind the new names.
Deleted together with the fallbacks."""

import os
import unittest
from unittest import mock

from jevelin.backends import classifier_credentials

NAMES = ("CLASSIFIER_API_KEY", "CLASSIFIER_URL", "TYPESAFE_API_KEY", "TYPESAFE_BASE_URL")


def _env(**values):
    clean = {k: v for k, v in os.environ.items() if k not in NAMES}
    clean.update(values)
    return mock.patch.dict(os.environ, clean, clear=True)


class LegacyEnvFallbacks(unittest.TestCase):
    def test_old_key_name_is_a_fallback(self):
        with _env():
            self.assertIsNone(classifier_credentials()[0])
        with _env(TYPESAFE_API_KEY="old-key"):
            self.assertEqual(classifier_credentials()[0], "old-key")
        with _env(TYPESAFE_API_KEY="old-key", CLASSIFIER_API_KEY="new-key"):
            self.assertEqual(classifier_credentials()[0], "new-key")

    def test_old_endpoint_name_is_a_fallback(self):
        with _env():
            self.assertIsNone(classifier_credentials()[1])
        with _env(TYPESAFE_BASE_URL="https://old.example.invalid"):
            self.assertEqual(classifier_credentials()[1], "https://old.example.invalid")
        with _env(TYPESAFE_BASE_URL="https://old.example.invalid", CLASSIFIER_URL="https://new.example.invalid"):
            self.assertEqual(classifier_credentials()[1], "https://new.example.invalid")


if __name__ == "__main__":
    unittest.main()
