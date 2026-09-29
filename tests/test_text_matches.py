"""Book4Source._text_matches: a GDZ scan must match the EXACT requested exercise."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from solver import Book4Source


class TestBook4SourceTextMatches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Empty config: no network at construction, and no dependence on the
        # real config.yaml.
        cls.source = Book4Source({})

    def test_neighbouring_exercise_on_the_same_page_does_not_match(self):
        self.assertFalse(
            self.source._text_matches("Стр.13 (6)", {"page": "13", "exercise": "7"})
        )

    def test_requested_exercise_matches(self):
        self.assertTrue(
            self.source._text_matches("Стр.13 (7)", {"page": "13", "exercise": "7"})
        )

    def test_match_is_per_exercise_not_per_page(self):
        # Same scan text as the first test; only the requested exercise differs.
        self.assertTrue(
            self.source._text_matches("Стр.13 (6)", {"page": "13", "exercise": "6"})
        )


if __name__ == "__main__":
    unittest.main()
