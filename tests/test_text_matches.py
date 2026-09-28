#!/usr/bin/env python3
"""Test for Book4Source._text_matches - verify exact exercise matching."""

import sys
sys.path.insert(0, 'src')

from solver import Book4Source
from solver import _load_config

config = _load_config()
source = Book4Source(config)

# Test case 1: Стр.13 (6) should NOT match for exercise='7'
test_text_1 = "Стр.13 (6)"
book_page_1 = {'page': '13', 'exercise': '7'}
result_1 = source._text_matches(test_text_1, book_page_1)
print(f"Test 1: text='{test_text_1}', book_page={book_page_1}")
print(f"  Expected: False, Got: {result_1}")
assert result_1 == False, f"FAILED: Expected False, got {result_1}"
print("  ✓ PASSED")

# Test case 2: Стр.13 (7) SHOULD match for exercise='7'
test_text_2 = "Стр.13 (7)"
book_page_2 = {'page': '13', 'exercise': '7'}
result_2 = source._text_matches(test_text_2, book_page_2)
print(f"\nTest 2: text='{test_text_2}', book_page={book_page_2}")
print(f"  Expected: True, Got: {result_2}")
assert result_2 == True, f"FAILED: Expected True, got {result_2}"
print("  ✓ PASSED")

# Test case 3: Стр.13 (6) should match for exercise='6'
test_text_3 = "Стр.13 (6)"
book_page_3 = {'page': '13', 'exercise': '6'}
result_3 = source._text_matches(test_text_3, book_page_3)
print(f"\nTest 3: text='{test_text_3}', book_page={book_page_3}")
print(f"  Expected: True, Got: {result_3}")
assert result_3 == True, f"FAILED: Expected True, got {result_3}"
print("  ✓ PASSED")

print("\n✓ ALL TESTS PASSED!")
