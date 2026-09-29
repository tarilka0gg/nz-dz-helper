import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nz_client import extract_book_page, extract_all_exercises
from solver import Task, classify_task, is_review_only, _text_has_keyword, _normalize_subject, _lookup_by_subject, _subject_may_have_figures


class TestExtractBookPage(unittest.TestCase):
    def test_empty_string_returns_none(self):
        self.assertIsNone(extract_book_page(""))

    def test_none_input_returns_none(self):
        self.assertIsNone(extract_book_page(None))

    def test_no_matching_patterns_returns_none(self):
        text = "Повторити тему"
        self.assertIsNone(extract_book_page(text))

    def test_page_range_with_en_dash(self):
        text = "Опрацювати матеріал на ст.3-10, напис. план"
        result = extract_book_page(text)
        self.assertEqual(result["page"], "3")
        self.assertEqual(result["page_end"], "10")
        self.assertIsNone(result["exercise"])
        self.assertIsNone(result["paragraph"])

    def test_page_range_with_em_dash(self):
        text = "Опрацювати матеріал на стор. 5 — 15"
        result = extract_book_page(text)
        self.assertEqual(result["page"], "5")
        self.assertEqual(result["page_end"], "15")

    def test_page_range_with_hyphen(self):
        text = "Опрацювати матеріал на сторінка 22-30"
        result = extract_book_page(text)
        self.assertEqual(result["page"], "22")
        self.assertEqual(result["page_end"], "30")

    def test_page_only(self):
        text = "Прочитати ст. 45"
        result = extract_book_page(text)
        self.assertEqual(result["page"], "45")
        self.assertIsNone(result["page_end"])

    def test_page_abbreviated_single_letter(self):
        result = extract_book_page("Прочитати с. 13")
        self.assertEqual(result["page"], "13")

    def test_russian_style_str_abbreviation_is_not_recognised(self):
        # Pins current behaviour, not a requirement: "Стр." (Russian-style)
        # matches none of the page patterns. The GDZ scan labels use it, but
        # no real homework text has been seen doing so.
        self.assertIsNone(extract_book_page("Стр.13"))

    def test_paragraph_with_section_symbol(self):
        text = "§ 5"
        result = extract_book_page(text)
        self.assertEqual(result["paragraph"], "5")
        self.assertIsNone(result["page"])
        self.assertIsNone(result["exercise"])

    def test_paragraph_with_paragraph_word(self):
        text = "Опрацювати параграф 12"
        result = extract_book_page(text)
        self.assertEqual(result["paragraph"], "12")
        self.assertIsNone(result["page"])
        self.assertIsNone(result["exercise"])

    def test_paragraph_with_section_and_number(self):
        text = "Прочитати §12"
        result = extract_book_page(text)
        self.assertEqual(result["paragraph"], "12")
        self.assertIsNone(result["page"])
        self.assertIsNone(result["exercise"])

    def test_paragraph_with_punkt_word(self):
        text = "Прочитати пункт 7"
        result = extract_book_page(text)
        self.assertEqual(result["paragraph"], "7")

    def test_exercise_with_vpraha(self):
        text = "розв'язати вправу 27"
        result = extract_book_page(text)
        self.assertEqual(result["exercise"], "27")

    def test_exercise_with_abbreviated_vpr(self):
        text = "впр. 5а"
        result = extract_book_page(text)
        self.assertEqual(result["exercise"], "5а")

    def test_exercise_with_number_sign(self):
        text = "№27.10"
        result = extract_book_page(text)
        self.assertEqual(result["exercise"], "27.10")

    def test_combined_page_and_exercise(self):
        text = "ст. 34, №15"
        result = extract_book_page(text)
        self.assertEqual(result["page"], "34")
        self.assertEqual(result["exercise"], "15")

    def test_combined_page_and_paragraph(self):
        text = "ст. 42, § 8"
        result = extract_book_page(text)
        self.assertEqual(result["page"], "42")
        self.assertEqual(result["paragraph"], "8")

    def test_combined_paragraph_and_exercise(self):
        text = "Прочитати параграф 9, вправа 3"
        result = extract_book_page(text)
        self.assertEqual(result["paragraph"], "9")
        self.assertEqual(result["exercise"], "3")

    def test_source_text_in_result(self):
        text = "ст. 10"
        result = extract_book_page(text)
        self.assertEqual(result["source_text"], text)


class TestExtractAllExercises(unittest.TestCase):
    def test_empty_string_returns_empty_list(self):
        self.assertEqual(extract_all_exercises(""), [])

    def test_none_input_returns_empty_list(self):
        self.assertEqual(extract_all_exercises(None), [])

    def test_no_exercise_patterns_returns_empty_list(self):
        text = "Повторити тему"
        self.assertEqual(extract_all_exercises(text), [])

    def test_single_exercise_with_vpraha(self):
        text = "вправа 5"
        self.assertEqual(extract_all_exercises(text), ["5"])

    def test_single_exercise_with_number_sign(self):
        text = "№27.10"
        self.assertEqual(extract_all_exercises(text), ["27.10"])

    def test_list_with_comma_separated_numbers(self):
        text = "розв'язати №27.10, 27.12, 27.13, 27.16"
        result = extract_all_exercises(text)
        self.assertEqual(result, ["27.10", "27.12", "27.13", "27.16"])

    def test_list_with_dot_space_separator(self):
        text = "розв'язати №27.10. 27.12, 27.13, 27.16"
        result = extract_all_exercises(text)
        self.assertEqual(result, ["27.10", "27.12", "27.13", "27.16"])

    def test_deduplication_preserves_order(self):
        text = "№5, 5, 6, 5"
        result = extract_all_exercises(text)
        self.assertEqual(result, ["5", "6"])

    def test_list_stops_at_a_non_number_separator(self):
        # The tail only continues over bare comma/dot-separated numbers; a
        # second keyword ("впр.") breaks the chain, so "3" is not collected.
        self.assertEqual(extract_all_exercises("вправа 5, №5, впр. 3, №3"), ["5"])

    def test_letter_suffixed_exercise(self):
        text = "вправа 5а"
        result = extract_all_exercises(text)
        self.assertEqual(result, ["5а"])

    def test_multiple_letter_suffixed_exercises(self):
        text = "вправа 5а, 5б, 5в"
        result = extract_all_exercises(text)
        self.assertEqual(result, ["5а", "5б", "5в"])


class TestTextHasKeyword(unittest.TestCase):
    def test_case_insensitive_match(self):
        self.assertTrue(_text_has_keyword("Написати ТВІР", ["твір"]))

    def test_no_match(self):
        self.assertFalse(_text_has_keyword("Написати твір", ["опис"]))

    def test_empty_keyword_list(self):
        self.assertFalse(_text_has_keyword("Написати твір", []))

    def test_substring_match(self):
        self.assertTrue(_text_has_keyword("Геометрія 9 клас", ["геометрі"]))

    def test_full_word_boundary_not_enforced(self):
        self.assertTrue(_text_has_keyword("Геометрія 9 клас", ["геометрі"]))


class TestNormalizeSubject(unittest.TestCase):
    def test_basic_normalization(self):
        self.assertEqual(_normalize_subject("Українська  література"), "українська література")

    def test_multiple_spaces(self):
        self.assertEqual(_normalize_subject("Українська    література"), "українська література")

    def test_trimming(self):
        self.assertEqual(_normalize_subject("  Українська література  "), "українська література")

    def test_case_insensitive(self):
        self.assertEqual(_normalize_subject("УкраїНСЬКА література"), "українська література")


class TestLookupBySubject(unittest.TestCase):
    def test_exact_key_match(self):
        mapping = {"Українська література": "ua_literature"}
        self.assertEqual(_lookup_by_subject(mapping, "Українська література"), "ua_literature")

    def test_double_space_in_key(self):
        mapping = {"Українська  література": "ua_literature"}
        self.assertEqual(_lookup_by_subject(mapping, "Українська література"), "ua_literature")

    def test_reverse_double_space_case(self):
        mapping = {"Українська література": "ua_literature"}
        self.assertEqual(_lookup_by_subject(mapping, "Українська  література"), "ua_literature")

    def test_case_insensitive(self):
        mapping = {"Українська література": "ua_literature"}
        self.assertEqual(_lookup_by_subject(mapping, "українська ЛІТЕРАТУРА"), "ua_literature")

    def test_no_match(self):
        mapping = {"Українська література": "ua_literature"}
        self.assertIsNone(_lookup_by_subject(mapping, "Математика"))

    def test_none_mapping(self):
        self.assertIsNone(_lookup_by_subject(None, "Математика"))

    def test_empty_mapping(self):
        self.assertIsNone(_lookup_by_subject({}, "Математика"))


class TestSubjectMayHaveFigures(unittest.TestCase):
    def test_geometriya_true(self):
        self.assertTrue(_subject_may_have_figures("Геометрія"))

    def test_matematika_true(self):
        self.assertTrue(_subject_may_have_figures("Математика"))

    def test_fizika_true(self):
        self.assertTrue(_subject_may_have_figures("Фізика"))

    def test_ukrainian_language_false(self):
        self.assertFalse(_subject_may_have_figures("Українська мова"))

    def test_history_false(self):
        self.assertFalse(_subject_may_have_figures("Історія"))

    def test_case_insensitive(self):
        self.assertTrue(_subject_may_have_figures("геометрія 9 клас"))


class TestIsReviewOnly(unittest.TestCase):
    def setUp(self):
        # Matching is a plain substring search with no inflection, so the real
        # config.yaml lists each case form explicitly ("контрольна",
        # "контрольну", ...) — mirrored here.
        self.config = {
            "write_task_keywords": ["твір", "есе"],
            "creative_subjects": ["Музичне мистецтво"],
            "review_keywords": ["повторити"],
            "assessment_keywords": ["контрольна", "контрольну", "самостійна"]
        }

    def test_unlisted_case_form_is_not_seen_as_assessment(self):
        # "контрольною" is not in this config's list and isn't a substring of
        # any listed form, so plain-substring matching misses it — which is
        # exactly why config.yaml enumerates the forms by hand.
        self.assertTrue(is_review_only("Повторити перед контрольною", config=self.config))

    def test_review_only_returns_true(self):
        homework_text = "Повторити матеріал ст. 34"
        self.assertTrue(is_review_only(homework_text, config=self.config))

    def test_review_with_assessment_returns_false(self):
        homework_text = "Повторити матеріал і написати контрольну роботу"
        self.assertFalse(is_review_only(homework_text, config=self.config))

    def test_no_review_keyword_returns_false(self):
        homework_text = "Виконати вправи 5-10"
        self.assertFalse(is_review_only(homework_text, config=self.config))

    def test_explicit_empty_config_is_not_replaced_by_the_real_config(self):
        # Regression: `config or _load_config()` treated {} as "no config"
        # and silently loaded the real config.yaml, so this returned True.
        self.assertFalse(is_review_only("Повторити матеріал", config={}))


class TestClassifyTask(unittest.TestCase):
    def setUp(self):
        self.config = {
            "write_task_keywords": ["твір", "есе"],
            "creative_subjects": ["Музичне мистецтво"],
            "review_keywords": ["повторити"],
            "assessment_keywords": ["контрольна", "самостійна"]
        }

    def test_write_task_wins_over_textbook(self):
        task = Task(subject="Українська література", homework_text="Написати твір-опис за §12")
        # Task.__post_init__ will call extract_book_page, which extracts "12" as paragraph
        result = classify_task(task, config=self.config)
        self.assertEqual(result, "write_task")

    def test_textbook_with_page_and_non_creative_subject(self):
        task = Task(subject="Математика", homework_text="Виконати вправи на ст. 34")
        result = classify_task(task, config=self.config)
        self.assertEqual(result, "textbook")

    def test_creative_with_creative_subject_and_page(self):
        task = Task(subject="Музичне мистецтво", homework_text="Прочитати ст. 15")
        result = classify_task(task, config=self.config)
        self.assertEqual(result, "creative")

    def test_creative_without_page(self):
        task = Task(subject="Українська література", homework_text="Обрати вірш для аналізу")
        self.assertIsNone(task.book_page)
        self.assertEqual(classify_task(task, config=self.config), "creative")

    def test_no_book_page_falls_through_to_creative_even_for_a_normal_subject(self):
        task = Task(subject="Історія", homework_text="Опрацювати тему")
        self.assertEqual(classify_task(task, config=self.config), "creative")

    def test_explicit_empty_config_is_not_replaced_by_the_real_config(self):
        # Regression, same root cause as the is_review_only one: with {} there
        # are no write_task keywords, so a "твір" text must NOT be write_task.
        task = Task(subject="Українська література", homework_text="Написати твір")
        self.assertEqual(classify_task(task, config={}), "creative")


if __name__ == "__main__":
    unittest.main()
