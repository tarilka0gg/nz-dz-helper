"""
Розв'язувач ДЗ по щоденнику nz.ua.

Бере уроки з JSON-виводу nz_client.py (get_diary_week), класифікує кожне ДЗ
як "textbook" (є конкретна сторінка/вправа з підручника — спершу шукаємо в
ГДЗ) чи "creative" (твір/есе — одразу LLM), і повертає розв'язок.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Optional

import yaml

from nz_client import extract_all_exercises, extract_book_page
from textbook_source import (
    DEFAULT_CACHE_DIR,
    download_textbook,
    extract_exercise_condition,
    extract_exercise_condition_with_figure,
    extract_page_range_text,
    extract_paragraph_text,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("solver")
timing_logger = logging.getLogger("solver.timing")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ -> корінь проєкту
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DB_PATH = PROJECT_ROOT / "nz_solver.db"

# Дефолтний таймаут openai-SDK — 600с (10 хв) + власні ретраї SDK поверх
# нього. Реальний живий тест (промт 44, 2026-09-08) зловив CloudflareProvider
# на 666с і 685с одного виклику — саме через відсутність цього timeout=,
# а не через паралелізацію: один зависший провайдер тримав слот
# _llm_semaphore (нижче) майже 11 хвилин, блокуючи все, що чекало за ним.
# 45с — з запасом на реальні LLM-пояснення (кілька секунд-десяток), але
# набагато швидший відмова->fallback, ніж дефолтні 600с. max_retries=1 —
# власний fallback-ланцюжок solver.py вже перебирає ІНШИХ провайдерів при
# невдачі, тож зайві внутрішні ретраї SDK на тому самому провайдері лише
# затягують час до failover'у.
_LLM_HTTP_TIMEOUT = 45.0
_LLM_HTTP_MAX_RETRIES = 1


# ---------------------------------------------------------------------- #
# Профілювання (промт 44: "чого так довго?" — timing-лог навколо кожного
# етапу /week, щоб мати РЕАЛЬНІ цифри замість здогадок про повільне місце).
# Формат навмисно єдиний ("[timing] <етап> subject=... : X.XXXs ...") —
# зручно для grep/агрегації після живого /week.
# ---------------------------------------------------------------------- #

from contextlib import contextmanager


@contextmanager
def _timed(stage: str, **fields):
    t0 = time.monotonic()
    try:
        yield
    finally:
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        timing_logger.info("[timing] %s %s: %.3fs", stage, extra, time.monotonic() - t0)


# Обмежує КІЛЬКІСТЬ ОДНОЧАСНИХ мережевих LLM-викликів (не GDZ/PDF) в усьому
# процесі — незалежно від того, скільки предметів/під-вправ розв'язується
# паралельно вище (telegram_bot.py -> _SOLVE_CONCURRENCY, і тепер ще
# _solve_multi_exercise нижче). Без цього сплеск паралельних Task (кожен
# предмет + кожен номер вправи в ньому) міг би одразу вдарити по
# rate-limit одного провайдера набагато швидше, ніж послідовна обробка —
# семафор просто ставить зайві виклики в чергу, а не валить їх помилкою.
# Розмір конфігурований (config.yaml -> llm.max_concurrent_calls), інакше
# 6 — створюється ОДИН РАЗ на весь процес (глобальний ресурс, не на
# LlmSolver-інстанс).
_llm_semaphore_lock = threading.Lock()
_llm_semaphore: Optional[threading.Semaphore] = None


def _get_llm_semaphore(config: dict) -> threading.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        with _llm_semaphore_lock:
            if _llm_semaphore is None:
                limit = int((config.get("llm") or {}).get("max_concurrent_calls", 6))
                _llm_semaphore = threading.Semaphore(max(1, limit))
                logger.info("Глобальний ліміт одночасних LLM-викликів: %d.", limit)
    return _llm_semaphore


def _call_provider(
    provider: "LlmProvider", system: str, prompt: str, config: dict, *, subject: str, provider_name: str
) -> str:
    """
    ЄДИНА точка виклику provider.complete() в усьому модулі (LlmSolver.solve()
    і _answers_match) — тут і семафор (обмежує одночасність), і timing-лог
    (скільки чекали семафора окремо від скільки тривав сам мережевий виклик).
    """
    sem = _get_llm_semaphore(config)
    t_wait0 = time.monotonic()
    sem.acquire()
    wait_s = time.monotonic() - t_wait0
    try:
        t0 = time.monotonic()
        try:
            return provider.complete(system, prompt)
        finally:
            elapsed = time.monotonic() - t0
            timing_logger.info(
                "[timing] llm_call subject=%r provider=%s: %.3fs (чекав семафор %.3fs)",
                subject, provider_name, elapsed, wait_s,
            )
    finally:
        sem.release()


class SolverError(Exception):
    """Базова помилка цього модуля."""


class LlmSolverError(SolverError):
    """Виклик LLM не вдався (мережа, авторизація, ліміти)."""


class BrowserSessionExpiredError(LlmSolverError):
    """Сесія браузерного профілю злетіла — бачу форму логіну замість чату."""


class TransientProviderError(LlmSolverError):
    """
    Провайдер тимчасово недоступний (rate-limit/quota АБО серверна
    помилка/перевантаження на боці самого провайдера) — НЕ проблема
    конфігурації. Спільний базовий клас для RateLimitError і
    ServerUnavailableError навмисно — LlmSolver.solve() ловить САМЕ цей
    тип (і обидва підкласи), щоб автоматично пробувати наступного
    провайдера з config.yaml -> llm.fallback_order. Інші помилки
    (авторизація, невідомий провайдер, зіпсована відповідь) не мають
    такого автоматичного failover'у — вони зазвичай означають реальну
    проблему конфігурації, яку тихий fallback лише замаскував би.
    """


class RateLimitError(TransientProviderError):
    """Провайдер повернув rate-limit/quota помилку (429/RESOURCE_EXHAUSTED)."""


class ServerUnavailableError(TransientProviderError):
    """
    Провайдер повернув серверну помилку/перевантаження (5xx, напр. Gemini
    503 "model is currently experiencing high demand") — на відміну від
    rate-limit це не "ти вичерпав ліміт", а "провайдеру зараз погано",
    але наслідок для solve() той самий: варто спробувати наступного в
    ланцюжку, а не одразу показувати помилку користувачу. Додано
    2026-09-07 після реального 503 від Gemini на живому /week, який
    раніше НЕ тригерив fallback (лише RateLimitError це робив).
    """


# ---------------------------------------------------------------------- #
# Конфіг
# ---------------------------------------------------------------------- #

def _load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        logger.warning("Файл конфігурації %s не знайдено — творчих предметів 0.", path)
        return {"creative_subjects": []}
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    data.setdefault("creative_subjects", [])
    return data


# ---------------------------------------------------------------------- #
# Task
# ---------------------------------------------------------------------- #

@dataclass
class Task:
    subject: str
    homework_text: str
    book_page: Optional[dict] = None
    exercise_source_text: str = ""

    def __post_init__(self) -> None:
        if self.book_page is None:
            self.book_page = extract_book_page(self.homework_text)
        if not self.exercise_source_text:
            self.exercise_source_text = self.homework_text


# ---------------------------------------------------------------------- #
# Класифікація
# ---------------------------------------------------------------------- #

def _text_has_keyword(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(kw.lower() in lowered for kw in keywords)


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_subject(name: str) -> str:
    return _WHITESPACE_RE.sub(" ", name).strip().casefold()


def _lookup_by_subject(mapping: Optional[dict], subject: str) -> Optional[str]:
    """
    Пошук значення для subject у config.yaml -> textbooks/gdz_sources.
    Спершу точний збіг ключа (швидкий шлях, як і раніше) — інакше
    нормалізований (зайві пробіли схлопнуті в один, регістр ігнорується).

    ВАЖЛИВО: назва предмета в щоденнику nz.ua й у config.yaml мають
    збігатись буквально — навіть один зайвий пробіл (як-от навмисний
    подвійний пробіл у "Українська  література", що відповідає реальному
    артефакту nz.ua) робив точний dict-lookup крихким: одна кома,
    скорочення чи інший регістр в назві предмета — і textbooks[subject]/
    gdz_sources[subject] мовчки повертали None, хоча запис у конфізі
    ФАКТИЧНО був (баг "немає книжки при наявному конфізі" — точна назва
    просто не збігалась символ-в-символ). Виправлено 2026-09-08.
    """
    if not mapping:
        return None
    if subject in mapping:
        return mapping[subject]
    target = _normalize_subject(subject)
    for key, value in mapping.items():
        if _normalize_subject(key) == target:
            return value
    return None


# Предмети, де умова реально посилається на рисунок/схему в підручнику —
# лише для них варто платити Vision-викликом на кожну вправу
# (extract_exercise_condition_with_figure нижче). Для решти предметів
# (мови, історія, суспільствознавство тощо) підручник текстовий, і
# Vision-виклик там лише витрачав квоту та повертав шумний опис на
# кшталт "На зображенні відсутні геометричні фігури…", який приліплювався
# до умови й плутав LLM — реальний баг, знайдений і виправлений 2026-09-09.
_FIGURE_RELEVANT_KEYWORDS = ("геометрі", "математик", "фізик")


def _subject_may_have_figures(subject: str) -> bool:
    lowered = subject.lower()
    return any(kw in lowered for kw in _FIGURE_RELEVANT_KEYWORDS)


def is_review_only(homework_text: str, config: Optional[dict] = None) -> bool:
    """
    True, якщо ДЗ — це просто "повторити конспект/матеріал" (config.yaml ->
    review_keywords), і текст НЕ згадує оцінювану роботу (config.yaml ->
    assessment_keywords). В такому разі розгорнуте пояснення від LLM зайве
    — сам факт "треба повторити" вже видно з тексту ДЗ, а якщо натомість
    йдеться про підготовку до контрольної/самостійної/тесту — це вже не
    "просто повторити", а привід дати чекліст тем (див. solve_task).
    """
    config = config or _load_config()
    review_keywords = config.get("review_keywords") or []
    assessment_keywords = config.get("assessment_keywords") or []

    if not _text_has_keyword(homework_text, review_keywords):
        return False
    return not _text_has_keyword(homework_text, assessment_keywords)


def classify_task(
    task: Task, config: Optional[dict] = None
) -> Literal["textbook", "creative", "write_task"]:
    """
    "write_task" — ДЗ явно просить написати готовий текст (твір/есе/розповідь
    — config.yaml -> write_task_keywords). Перевіряється ПЕРШИМ, незалежно
    від предмету/наявності сторінки підручника: "Написати твір-опис за §12"
    — це прохання написати текст, а не вправа з підручника для розв'язання.

    Інакше — "textbook", якщо в ДЗ знайдено сторінку/вправу (book_page не
    None) і предмет не в списку творчих (config.yaml -> creative_subjects).
    Інакше — "creative".
    """
    config = config or _load_config()

    write_task_keywords = config.get("write_task_keywords") or []
    if _text_has_keyword(task.homework_text, write_task_keywords):
        return "write_task"

    creative_subjects = set(config.get("creative_subjects", []))
    if task.book_page and task.subject not in creative_subjects:
        return "textbook"
    return "creative"


# ---------------------------------------------------------------------- #
# ГДЗ-джерело (заглушка)
# ---------------------------------------------------------------------- #

class GdzSource(ABC):
    """
    Базовий клас для скрейперів конкретних ГДЗ-сайтів.

    Реальні реалізації (під конкретні підручники/сайти) допишемо окремо,
    коли визначимось з джерелами — цей клас лише задає інтерфейс.
    """

    @abstractmethod
    def search(self, subject: str, book_page: dict) -> Optional[dict]:
        """
        Повертає {"raw_answer": str | None, "source_image_url": str | None},
        або None якщо взагалі нічого не знайдено (сторінка/номер відсутні,
        404 тощо). raw_answer заповнений одразу, якщо на сторінці був
        реальний текст; якщо відповідь — скан, raw_answer=None і
        source_image_url вказує на зображення (OCR робить викликач,
        solve_task, через _ocr_scan — див. нижче).
        """


class NoOpGdzSource(GdzSource):
    """Заглушка за замовчуванням — завжди None, поки немає реальних скрейперів."""

    def search(self, subject: str, book_page: dict) -> Optional[dict]:
        return None


class Book4Source(GdzSource):
    """
    Скрепер під 4book.org.

    Навігація на сайті трирівнева: сторінка книги -> сторінка теми/параграфа
    (номери/діапазони в тексті посилань, напр. "№ 1.1 - 2.30" або
    "Стр.186 (3)") -> кінцева сторінка з відповіддю. Обидва верхні рівні
    використовують звичайні <a href="..."> в межах шляху книги — окремого
    стабільного класу для навігації немає на першому рівні, тому фільтрація
    йде за префіксом URL книги, а номер/діапазон береться з повного тексту
    посилання (включно з вкладеними <span>).

    ВАЖЛИВО (перевірено наживо на 3 предметах — алгебра, українська мова,
    історія україни): кінцева відповідь на цьому сайті ЗАВЖДИ скановане
    зображення (<img id="imgZoom"> в .img-content), а не текст. search()
    повертає {"raw_answer": None, "source_image_url": <url>} в цьому
    випадку — OCR і звірку з LLM робить solve_task() (через _ocr_scan).
    Метод текстової екстракції лишений на випадок сторінок, де відповідь
    таки текстова (рідкісний випадок, не спостерігався в тесті) — тоді
    raw_answer заповнений одразу і source_image_url=None.
    """

    BASE_URL = "https://4book.org"
    _USER_AGENT = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )

    _PAGE_EXERCISE_RE = re.compile(r"стр\.?\s*(\d+)\s*\((\d+)\)", re.IGNORECASE)
    _RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)")
    _SINGLE_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")

    def __init__(self, config: Optional[dict] = None, timeout: int = 10):
        self.config = config if config is not None else _load_config()
        self.timeout = timeout
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update({"User-Agent": self._USER_AGENT})
        return self._session

    def search(self, subject: str, book_page: dict) -> Optional[dict]:
        gdz_sources = self.config.get("gdz_sources") or {}
        book_url = _lookup_by_subject(gdz_sources, subject)
        if not book_url:
            logger.info(
                "Немає gdz_sources[%r] в config.yaml — звірка з ГДЗ пропускається "
                "для цього предмету.",
                subject,
            )
            return None

        if not book_page or not (book_page.get("page") or book_page.get("exercise")):
            return None

        try:
            return self._search_exercise(book_url, book_page)
        except Exception as exc:  # мережа/парсинг — ГДЗ ніколи не має валити весь solve_task
            logger.info(
                "Book4Source: помилка при пошуку '%s' у %s: %s", book_page, book_url, exc
            )
            return None

    def _fetch(self, url: str) -> Optional[str]:
        import requests

        session = self._get_session()
        try:
            resp = session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            logger.info("Book4Source: не вдалось завантажити %s: %s", url, exc)
            return None
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.text

    def _search_exercise(self, book_url: str, book_page: dict) -> Optional[dict]:
        if not book_url.startswith("http"):
            book_url = self.BASE_URL + ("" if book_url.startswith("/") else "/") + book_url
        book_url = book_url.rstrip("/")

        html = self._fetch(book_url)
        if html is None:
            return None

        mid_url = self._find_matching_link(html, book_url, book_url, book_page)
        if mid_url is None:
            return None

        html2 = self._fetch(mid_url)
        if html2 is None:
            return None

        # Другий рівень може вже бути кінцевою сторінкою з відповіддю, або
        # ще одним списком (найчастіше — саме списком з конкретнішими
        # номерами). Пробуємо знайти ще точніше посилання; якщо нема — це
        # вже кінцева сторінка.
        leaf_url = self._find_matching_link(html2, mid_url, book_url, book_page)
        target_html = html2
        if leaf_url and leaf_url != mid_url:
            fetched = self._fetch(leaf_url)
            if fetched is not None:
                target_html = fetched

        return self._extract_text_answer(target_html)

    def _find_matching_link(
        self, html: str, base_url: str, book_url_prefix: str, book_page: dict
    ) -> Optional[str]:
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = urljoin(base_url, a["href"])
            if not href.startswith(book_url_prefix):
                continue
            text = a.get_text(" ", strip=True)
            
            # Skip "garbage" links that lead to main pages or subject lists
            # These don't contain specific exercise/page info and will match incorrectly
            if self._is_garbage_link(text, href):
                logger.debug("=== _find_matching_link: SKIP GARBAGE LINK ===")
                logger.debug("text: %s, href: %s", text, href)
                continue
            
            logger.debug("=== _find_matching_link проверка посилання ===")
            logger.debug("href: %s", href)
            logger.debug("text: %s", text)
            if self._text_matches(text, book_page):
                logger.debug("=== _find_matching_link: ЗНАЙДЕНО ВІДПОВІДНЕ ПОСИЛАННЯ ===")
                return href
        logger.debug("=== _find_matching_link: ЖОДНОГО ЗБІГУ НЕ ЗНАЙДЕНО ===")
        return None

    @staticmethod
    def _is_garbage_link(text: str, href: str) -> bool:
        """
        Перевіряє, чи посилання є "мусорним" (веде на головну сторінку, список підручників тощо).
        Такі посилання не містять конкретної інформації про вправу/сторінку і можуть неправильно збігтися.
        """
        # Check if href is a main page or navigation
        if href.endswith(("/", "/index.html", "/main.html")):
            return True
        
        # Check if text looks like navigation/menu
        garbage_patterns = [
            r"^Підручники\s*\d+\s*клас.*$",
            r"^ГДЗ\s*\d+\s*клас.*$",
            r"^[Пп]ро\s*[Пп]роект$",
            r"^[Яя]\s*[Уу]\s*[Ссвіті]$",
            r"^[Аа]нглійська\s*[Мм]ова.*$",
            r"^[Бб]іологія$",
            r"^[Зз]арубіжна\s*[Лл]ітература$",
            r"^[Уу]країнська\s*[Лл]ітература$",
            r"^[Уу]країнська\s*[Мм]ова$",
            r"^[Фф]ранцузька\s*[Мм]ова$",
            r"^[Яя]\s*[Уу]\s*[Ссвіті]$",
            r"^[ДдППАА].*$",
            r"^[ЗзНнОо]$",
            r"^[Рр]еферати$",
            r"^[Аа]втори$",
            r"^[Кк]онтакти$",
            r"^[Пп]олітика\s*[Кк]онфідентності$",
        ]
        import re
        for pat in garbage_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return True
        
        return False

    @staticmethod
    def _parse_ordinal(text: str) -> Optional[tuple]:
        """
        "1.13" -> (1, 13); "12" -> (12, None). Параграф.номер — це ПОРЯДКОВА
        пара, не десятковий дріб: вправа "1.13" йде ПІСЛЯ "1.2" (13 > 2 як
        ціле), хоча як float 1.13 < 1.2 — тому порівнюємо парами (major,
        minor), не float.
        """
        text = text.strip()
        if "." in text:
            major, _, minor = text.partition(".")
            try:
                return (int(major), int(minor))
            except ValueError:
                return None
        try:
            return (int(text), None)
        except ValueError:
            return None

    @staticmethod
    def _find_paragraph_match(text: str, target_paragraph: str) -> bool:
        """
        Перевіряє чи текст містить номер параграфа.
        Наприклад: "2. Типи словників" містить paragraph=2.
        Витягує перше число з тексту і порівнює з target_paragraph.
        """
        # Спробуємо знайти номер параграфа на початку тексту (наприклад "2.")
        first_num_match = re.match(r'^(\d+)\.', text.strip())
        if first_num_match:
            found_paragraph = first_num_match.group(1)
            logger.debug("Found paragraph number at start: %s (target: %s)", found_paragraph, target_paragraph)
            if found_paragraph == target_paragraph:
                return True
        
        # Якщо не знайдено на початку, шукаємо в тексті патерн "N." де N - номер
        # Але не після першого слова (це може бути номер вправи в середині)
        # Шукаємо патерн "N." де N - число на початку або після " № " або після " - "
        paragraph_patterns = [
            re.compile(r'^(\d+)\.', re.IGNORECASE),
            re.compile(r' №\s*(\d+)\.', re.IGNORECASE),
            re.compile(r' -\s*(\d+)\.', re.IGNORECASE),
        ]
        for pat in paragraph_patterns:
            m = pat.search(text)
            if m:
                found_paragraph = m.group(1)
                logger.debug("Found paragraph number via pattern %s: %s (target: %s)", pat.pattern, found_paragraph, target_paragraph)
                if found_paragraph == target_paragraph:
                    return True
        
        return False

    @staticmethod
    def _ordinal_key(ordinal: tuple) -> tuple:
        major, minor = ordinal
        return (major, minor if minor is not None else -1)

    def _text_matches(self, text: str, book_page: dict) -> bool:
        page = book_page.get("page")
        exercise = book_page.get("exercise")
        paragraph = book_page.get("paragraph")
        
        # DEBUG: логування вхідних даних
        logger.debug(
            "=== _text_matches DEBUG ===\n"
            "Текст з посилання: %s\n"
            "Ціль page: %s, exercise: %s, paragraph: %s",
            text, page, exercise, paragraph
        )
        
        # Якщо заданий paragraph, спершу перевіряємо його (номер параграфа/теми)
        logger.debug("_text_matches: text=%s, book_page=%s", text, book_page)
        if paragraph:
            paragraph_match = self._find_paragraph_match(text, str(paragraph))
            logger.debug("Paragraph match check: %s", paragraph_match)
            if paragraph_match:
                # Якщо знайдено параграф, це добре, але не обов'язково.
                # Наприклад, ДЗ може бути "Опрацювати §2, вик. впр.7(ст.13)",
                # але вправа 7 може бути в іншому параграфі на тій самій сторінці.
                logger.debug("Paragraph match found, but this is not mandatory")
            # Якщо paragraph не знайдено, не відкидаємо скан — просто продовжуємо перевірку page/exercise
            logger.debug("Paragraph not found, continuing with page/exercise check")
        
        # СПЕРШУ перевіряємо патерн "стр.X (Y)" який означає "сторінка X, вправа Y"
        logger.debug("Trying PAGE_EXERCISE_RE on: %s", text)
        m = self._PAGE_EXERCISE_RE.search(text)
        page_exercise_match = False
        found_page = None
        found_exercise = None
        if m:
            found_page = int(m.group(1))
            found_exercise = m.group(2)  # номер в дужках — це НОМЕР ВПРАВИ!
            logger.debug("PAGE_EXERCISE_RE match: page=%s (з regex), exercise=%s (з regex)", found_page, found_exercise)
            
            # Якщо знайдено номер сторінки в дужках, вважати його як номер вправи
            # Наприклад "стр.7 (6)" означає "сторінка 7, вправа 6"
            # Але якщо page=13, це означає що шукаємо сторінку 13, а не вправу 13
            if page:
                if found_page == int(page):
                    # Якщо page збігається, вважати номер в дужках як номер вправи
                    logger.debug("Page match: found_page=%s == page=%s", found_page, page)
                    page_exercise_match = True
                    if exercise is None:
                        logger.debug("=== _text_matches: MATCH (стр.(N) page only) ===")
                        return True
                    exercise_match = str(found_exercise) == str(exercise).strip()
                    logger.debug("Exercise match (з дужок): %s == %s = %s", found_exercise, exercise, exercise_match)
                    if exercise_match:
                        logger.debug("=== _text_matches: MATCH (стр.(N)) ===")
                        return True
                else:
                    # Page не збігається, це НЕ match для PAGE_EXERCISE_RE
                    logger.debug("Page mismatch: found_page=%s != page=%s", found_page, page)
                    page_exercise_match = False
            elif not page:
                # Якщо page не заданий, вважати номер в дужках як номер вправи
                logger.debug("Page not specified, using exercise from parens: %s", found_exercise)
                if exercise is None or str(found_exercise) == str(exercise).strip():
                    logger.debug("=== _text_matches: MATCH (стр.(N) exercise only) ===")
                    return True
                page_exercise_match = True
        
        # DEBUG: логування стану після перевірки PAGE_EXERCISE_RE
        logger.debug(
            "=== _text_matches PAGE_EXERCISE_RE RESULT ===\n"
            "m=%s, found_page=%s, found_exercise=%s, page=%s, exercise=%s\n"
            "page_exercise_match=%s",
            m, found_page, found_exercise, page, exercise, page_exercise_match
        )
        
        # Якщо page заданий і PAGE_EXERCISE_RE знайшов page, але page не збіглося,
        # то перевіряємо чи це page range. Якщо немає "Стор.", то це НЕ match.
        if page and m and found_page != int(page):
            page_range_re_early = re.compile(r'Стор\.?\s*(\d+(?:\.\d+)?)\s*[-–—]\s*(?:Стор\.?\s*)?(\d+(?:\.\d+)?)', re.IGNORECASE)
            is_page_range_early = page_range_re_early.search(text) is not None
            if not is_page_range_early:
                logger.debug("PAGE_EXERCISE_RE page mismatch and not page range, skipping")
                return False

        
        # DEBUG: логування стану після перевірки PAGE_EXERCISE_RE
        logger.debug(
            "=== _text_matches PAGE_EXERCISE_RE RESULT ===\n"
            "m=%s, found_page=%s, found_exercise=%s, page=%s, exercise=%s\n"
            "page_exercise_match=%s",
            m, found_page if m else None, found_exercise if m else None, page, exercise, page_exercise_match
        )
        
        if exercise:
            target = self._parse_ordinal(str(exercise))
            logger.debug("Parsed target exercise ordinal: %s", target)
            if target is not None:
                tk = self._ordinal_key(target)
                logger.debug("Target key: %s", tk)
                
                # СПЕРШУ перевіряємо чи це діапазон сторінок (Стор.X - Стор.Y) або вправ
                # Якщо є "Стор." або "ст.", то це сторінки, а не вправи
                # Підтримуємо: "Стор. 10 - 17" або "Стор. 10 - Стор. 17"
                page_range_re = re.compile(r'Стор\.?\s*(\d+(?:\.\d+)?)\s*[-–—]\s*(?:Стор\.?\s*)?(\d+(?:\.\d+)?)', re.IGNORECASE)
                is_page_range = page_range_re.search(text) is not None
                
                logger.debug("Is page range (contains Стор.): %s", is_page_range)
                
                rm = self._RANGE_RE.search(text)
                if rm:
                    logger.debug("RANGE_RE matched groups: %s-%s", rm.group(1), rm.group(2))
                    # Якщо це page range, парсимо як plain numbers (page), інакше як exercise ordinals
                    if is_page_range and page:
                        lo_str, hi_str = rm.group(1), rm.group(2)
                        try:
                            lo, hi = int(lo_str), int(hi_str)
                            logger.debug("Page range parsed: lo=%s, hi=%s", lo, hi)
                            page_val = int(page)
                            in_range = lo <= page_val <= hi
                            logger.debug("Page range check: %s <= %s <= %s = %s", lo, page_val, hi, in_range)
                            if in_range:
                                logger.debug("=== _text_matches: MATCH (PAGE RANGE) ===")
                                return True
                            else:
                                # Page range не збіглося, це НЕ match
                                logger.debug("Page range mismatch")
                        except ValueError:
                            logger.debug("Page range parse failed, falling back to exercise range logic")
                            lo, hi = self._parse_ordinal(rm.group(1)), self._parse_ordinal(rm.group(2))
                            if lo is None or hi is None:
                                lo, hi = self._parse_ordinal(rm.group(1)), self._parse_ordinal(rm.group(2))
                    else:
                        lo, hi = self._parse_ordinal(rm.group(1)), self._parse_ordinal(rm.group(2))
                    
                    # Якщо це exercise range (або page range з помилкою парсингу), продовжуємо з ordinal
                    # Але якщо page range успішно парсився і match, ми вже повернули True
                    # Тут лише exercise range або page range з помилкою
                    # Якщо lo, hi це int (page range), то пропускаємо ordinal key conversion
                    if lo is not None and hi is not None:
                        if isinstance(lo, tuple) and isinstance(hi, tuple):
                            lo_key = self._ordinal_key(lo)
                            hi_key = self._ordinal_key(hi)
                            logger.debug("lo_key=%s, hi_key=%s", lo_key, hi_key)
                            # Якщо це діапазон вправ, порівнюємо з exercise
                            in_range = lo_key <= tk <= hi_key
                            logger.debug("Range check: %s <= %s <= %s = %s", lo_key, tk, hi_key, in_range)
                            if in_range:
                                logger.debug("=== _text_matches: MATCH (RANGE) ===")
                                return True
                        else:
                            # lo, hi це int (page range), але це вже було перевірено вище
                            # Якщо ми тут, то page range match не відбувся (інший діапазон)
                            logger.debug("Page range values are int, but check already failed above")
                else:
                    # ВИПРАВЛЕНИЙ КОД: для "стр.X (Y)" вважати Y як номер вправи, а не X
                    # Перевіряємо чи є в тексті патерн "стр.X (Y)" — якщо є, використовуємо Y
                    single = None
                    pe_match = self._PAGE_EXERCISE_RE.search(text)
                    if pe_match:
                        found_exercise = pe_match.group(2)
                        single = self._parse_ordinal(found_exercise)
                        logger.debug("PAGE_EXERCISE_RE found exercise in parens: %s", single)
                    else:
                        sm = self._SINGLE_NUM_RE.search(text)
                        if sm:
                            single = self._parse_ordinal(sm.group(1))
                            logger.debug("SINGLE_NUM_RE match: %s", single)
                    if single is not None:
                        exact_match = single == target
                        logger.debug("Exact match: %s == %s = %s", single, target, exact_match)
                        if exact_match:
                            logger.debug("=== _text_matches: MATCH (SINGLE) ===")
                            return True
        logger.debug("=== _text_matches: NO MATCH ===")
        return False

    def _extract_text_answer(self, html: str) -> Optional[dict]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        img = soup.select_one(".img-content img, #imgZoom")
        if img and img.get("src"):
            image_url = img["src"]
            logger.info(
                "Book4Source: відповідь на сторінці — скановане зображення "
                "(%s). OCR/звірку зробить solve_task().",
                image_url,
            )
            return {"raw_answer": None, "source_image_url": image_url}

        content = soup.select_one(".task-block, article, .entry-content") or soup
        parts = [
            el.get_text(strip=True)
            for el in content.find_all(["p", "li"])
            if el.get_text(strip=True)
        ]
        combined = "\n".join(parts)
        if len(combined) < 20:
            return None
        return {"raw_answer": combined, "source_image_url": None}


class FreeGdzSource(GdzSource):
    """
    Заготовка під freegdz.com — НЕ РЕАЛІЗОВАНО.

    Перевірено наживо (2026-09-05, алгебра Мерзляк 10 клас): та сама
    ситуація, що й з 4book.org — кінцева відповідь це <img
    src=".../answerbookcontents/.../N-XXXXXX.jpg"> без будь-якого
    текстового супроводу. Навігація трирівнева (предмет -> книга ->
    параграф -> vprava<N>-<id>), простіша за 4book.org (номер вправи прямо
    в URL останнього рівня: ".../vprava1-325683"), тож коли знадобиться —
    писати за зразком Book4Source._search_exercise, просто з іншим
    URL-патерном.
    """

    def search(self, subject: str, book_page: dict) -> Optional[dict]:
        return None


class VshkoleSource(GdzSource):
    """
    Заготовка під vshkole.com — НЕ РЕАЛІЗОВАНО.

    Перевірено наживо (2026-09-05, алгебра Мерзляк 10 клас): ідентична
    ситуація — <img> в .img-content зі скановою відповіддю, той самий
    UI (zoom-plus/zoom-minus), що й на 4book.org — схоже на спільну
    платформу/шаблон. Навігація: предмет -> книга -> параграф ->
    .../<номер вправи> (номер просто останнім сегментом шляху, без
    префікса на кшталт "vprava"/"page-").
    """

    def search(self, subject: str, book_page: dict) -> Optional[dict]:
        return None


# ---------------------------------------------------------------------- #
# LLM-провайдери
# ---------------------------------------------------------------------- #

_EXPLAIN_SYSTEM_EXACT = """\
Ти — репетитор, який допомагає учню 10 класу з домашнім завданням через Telegram.

Обсяг відповіді:
- Для точних наук (математика, фізика, хімія) — розгорнуте покрокове
  пояснення з розрахунками (3–6 речень або більше, якщо потрібні
  розрахунки).
- Для гуманітарних предметів (історія, мови, література) — коротко:
  3–5 речень без вступів і висновків.

Стиль:
- НІКОЛИ не розігруй діалог "Я (репетитор):" / "Учень:" — звертайся до
  учня прямо, без сценки.
- НІКОЛИ не використовуй Markdown-таблиці (рядки з "|"). Замість таблиці —
  звичайний список ("•" або "1.", "2.").
- Для точних наук: покрокові розрахунки, формулки, висновки.
- Для гуманітарних: конкретні факти/дати/терміни, без пафосу.

Розмітка виводу — ПРОСТИЙ MARKDOWN (не HTML, теги на кшталт <b> тут писати
НЕ треба — конвертацію в Telegram-розмітку робить окремий код, не ти):
- **жирний текст** для ключових термінів/висновків.
- _курсив_ для другорядного наголосу (рідко, за потреби).
- `код`/формули у зворотних лапках, якщо доречно.
- Списки — звичайним текстом: кожен пункт на новому рядку, що починається
  з "•" або "1.", "2." — БЕЗ Markdown-таблиць (рядків з "|") і без ### заголовків.

Відповідай українською мовою.
"""

_EXPLAIN_SYSTEM_HUMANITIES = """\
Ти — репетитор, який допомагає учню 10 класу з домашнім завданням через Telegram.

ВАЖЛИВО про поле "Завдання:" нижче в повідомленні користувача: якщо там
є розгорнутий текст (умова вправи, речення для аналізу, уривок,
питання тощо) — це РЕАЛЬНИЙ текст з підручника, і ти МАЄШ використати
САМЕ його для відповіді (перекажи суть, виконай завдання на основі
цього тексту). НІКОЛИ не пиши "не маю тексту параграфа", якщо в
"Завдання:" фактично є такий текст — навіть коли сама фраза ДЗ у
щоденнику звучить організаційно ("опрацювати §N", "вивчити" тощо):
дивись на вміст поля "Завдання:", а не на формулювання зі щоденника.

Обсяг відповіді:
- Коротко: 3–5 речень без вступів, висновків, пафосу.
- Лише якщо "Завдання:" містить ЛИШЕ короткий організаційний запис БЕЗ
  жодного розгорнутого тексту (напр. рівно "Повторити конспект" і
  більше нічого) — тоді скажи: "Не маю тексту параграфа" і дай
  загальний метод роботи (2–3 речення). НІКОЛИ не вигадуй конкретні
  терміни/факти, яких немає в наданому тексті.

Стиль:
- НІКОЛИ не розігруй діалог "Я (репетитор):" / "Учень:" — звертайся до
  учня прямо, без сценки.
- НІКОЛИ не використовуй Markdown-таблиці (рядки з "|"). Замість таблиці —
  звичайний список ("•" або "1.", "2.").
- Конкретні факти/дати/терміни, без лірики.

Розмітка виводу — ПРОСТИЙ MARKDOWN (не HTML):
- **жирний текст** для ключових термінів/висновків.
- _курсив_ для другорядного наголосу (рідко).
- Списки — звичайним текстом: "•" або "1.", "2." на новому рядку.

Відповідай українською мовою.
"""

_ANSWER_SYSTEM = """\
Ти допомагаєш учню 10 класу з домашнім завданням через Telegram.
Дай розв'язок задачі з дуже коротким поясненням ходу думок (1–3 речення) —
не гола відповідь, але й не урок.

Розмітка виводу — ПРОСТИЙ MARKDOWN (не HTML, теги на кшталт <b> тут писати
НЕ треба — конвертацію в Telegram-розмітку робить окремий код, не ти):
- **жирний текст** для ключових термінів/висновків, _курсив_ рідко, `код`
  для формул, якщо доречно.
- Списки — звичайним текстом ("•" або "1.", "2." на новому рядку).
- Без Markdown-таблиць, без ### заголовків, без рольових діалогів
  "Я (репетитор):"/"Учень:".
Відповідай українською мовою.
"""

_WRITE_SYSTEM = """\
Ти пишеш ГОТОВИЙ текст шкільного твору/есе для учня 10 класу — не план, не
поради як писати, а сам зв'язний текст.

Стиль:
- Пиши ФАКТИЧНО і по суті, без емоційних зворотів ("надихає", "дивовижно",
  "невід'ємна частина життя", "щодня я помічаю") — це не есе-роздум про
  почуття, а виклад фактів і понять.
- Уникай пафосних узагальнень і високого стилю. Прості розповідні речення.
- Структура кожної частини: конкретний факт/явище -> яке поняття (фізичне,
  історичне тощо — залежно від предмету) за ним стоїть -> короткий приклад.
  БЕЗ ліричних вступів і без висновків "про майбутнє людства"/загального
  значення теми для життя.

Обсяг:
- Якщо в завданні вказана КОНКРЕТНА кількість слів (напр. "200-300 слів")
  — дотримуйся її ТОЧНО, перевищення верхньої межі не більш ніж на 10-15%.
- Якщо кількість слів НЕ вказана явно (напр. просто "коротенький твір-опис",
  без цифр) — за замовчуванням 120–180 слів. Це реалістичний обсяг для
  шкільного домашнього твору, а не есе на кілька сторінок.

КАТЕГОРИЧНО ЗАБОРОНЕНО:
- Видавати план/схему/поради "як писати" ("Крок 1: ... Крок 2: ...",
  "спочатку зроби вступ, потім опиши..."). Учню потрібен ГОТОВИЙ ТЕКСТ,
  а не інструкція як його написати.
- Перераховувати більше 3-4 прикладів/явищ в одному творі — краще розкрити
  МЕНШЕ прикладів трохи детальніше, ніж дати список із 7-8 пунктів поспіль
  (це й так неминуче ламає обсяг у 120-180 слів).
- Метакоментарі до/після самого тексту ("Ось твір на тему...", "Сподіваюсь,
  це допоможе", назва твору окремим рядком типу "Твір: ...") — почни одразу
  з тексту, закінчи одразу текстом, без обрамлення.
- Рольові діалоги "Я (репетитор):" / "Учень:".

Розмітка виводу — простий Markdown (не HTML): суцільний зв'язний текст
(абзаци), **жирний**/списки тут НЕ доречні. Без ### заголовків.

Відповідай українською мовою.
"""

_ASSESSMENT_PREP_ADDENDUM = """

ОСОБЛИВИЙ РЕЖИМ: текст ДЗ згадує підготовку до оцінюваної роботи (контрольна/
самостійна/тест/залік/атестація) — це НЕ звичайне ДЗ з конкретним завданням.
Замість пояснення дай СТИСЛИЙ ЧЕКЛІСТ тем/понять, які варто повторити перед
цією роботою, пунктами ("•" або "1.", "2."), без розгорнутого пояснення
кожної теми.
"""


class LlmProvider(ABC):
    """Спільний інтерфейс для будь-якого LLM-бекенду."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Повертає текст відповіді, або кидає LlmSolverError."""


class GroqProvider(LlmProvider):
    """OpenAI-сумісний SDK, base_url на api.groq.com (безкоштовний тір без картки)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-120b",
        max_tokens: int = 4000,
    ):
        try:
            import openai
        except ImportError as exc:
            raise LlmSolverError(
                "Пакет 'openai' не встановлено (pip install openai)."
            ) from exc

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise LlmSolverError(
                "Немає GROQ_API_KEY (ані в аргументі, ані в .env/оточенні)."
            )

        self._openai = openai
        self.client = openai.OpenAI(
            api_key=key,
            base_url="https://api.groq.com/openai/v1",
            timeout=_LLM_HTTP_TIMEOUT,
            max_retries=_LLM_HTTP_MAX_RETRIES,
        )
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        openai = self._openai
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except openai.RateLimitError as exc:
            raise RateLimitError(
                f"Денний ліміт/rate-limit вичерпано для 'groq'. Деталі: {exc}"
            ) from exc
        except openai.AuthenticationError as exc:
            raise LlmSolverError(f"Невірний GROQ_API_KEY: {exc}") from exc
        except openai.InternalServerError as exc:
            raise ServerUnavailableError(f"Groq: серверна помилка/перевантаження: {exc}") from exc
        except openai.APIStatusError as exc:
            raise LlmSolverError(f"Groq API повернув помилку: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise LlmSolverError(f"Не вдалось з'єднатись з Groq API: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice and choice.message else None
        if not text:
            raise LlmSolverError("Groq API повернув відповідь без тексту.")
        if choice.finish_reason == "length":
            logger.warning("Groq: відповідь обірвана лімітом max_tokens=%d.", self.max_tokens)
            text += (
                "\n\n⚠️ Відповідь обірвана лімітом довжини — модель не встигла "
                "закінчити думку. Онов max_tokens в config.yaml/GroqProvider, "
                "якщо це трапляється часто."
            )
        return text


class GeminiProvider(LlmProvider):
    """google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 4000,
    ):
        try:
            from google import genai
            from google.genai import errors as genai_errors
            from google.genai import types as genai_types
        except ImportError as exc:
            raise LlmSolverError(
                "Пакет 'google-genai' не встановлено (pip install google-genai)."
            ) from exc

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LlmSolverError(
                "Немає GEMINI_API_KEY (ані в аргументі, ані в .env/оточенні)."
            )

        self._genai_errors = genai_errors
        self._genai_types = genai_types
        self.client = genai.Client(
            api_key=key,
            http_options=genai_types.HttpOptions(timeout=int(_LLM_HTTP_TIMEOUT * 1000)),
        )
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        genai_types = self._genai_types
        genai_errors = self._genai_errors
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=self.max_tokens,
                ),
            )
        except genai_errors.ClientError as exc:
            if getattr(exc, "code", None) == 429:
                raise RateLimitError(
                    f"Денний ліміт/rate-limit вичерпано для 'gemini'. Деталі: {exc}"
                ) from exc
            raise LlmSolverError(
                f"Gemini API: помилка клієнта (перевір GEMINI_API_KEY/квоти): {exc}"
            ) from exc
        except genai_errors.ServerError as exc:
            # Саме цей шлях (напр. 503 "model is currently experiencing
            # high demand") раніше НЕ тригерив fallback — тільки 429 через
            # ClientError вище це робив. Реальний живий випадок
            # (2026-09-07, /week, Геометрія) показав, що 503 теж має
            # переходити на наступного провайдера, а не одразу падати.
            raise ServerUnavailableError(f"Gemini API: серверна помилка/перевантаження: {exc}") from exc
        except genai_errors.APIError as exc:
            raise LlmSolverError(f"Gemini API повернув помилку: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise LlmSolverError("Gemini API повернув відповідь без тексту.")

        candidates = getattr(response, "candidates", None) or []
        finish_reason = str(candidates[0].finish_reason) if candidates else ""
        if "MAX_TOKENS" in finish_reason:
            logger.warning("Gemini: відповідь обірвана лімітом max_tokens=%d.", self.max_tokens)
            text += (
                "\n\n⚠️ Відповідь обірвана лімітом довжини — модель не встигла "
                "закінчити думку. Онов max_tokens в config.yaml/GeminiProvider, "
                "якщо це трапляється часто."
            )
        return text


class OvhCloudProvider(LlmProvider):
    """
    base_url на oai.endpoints.kepler.ai.cloud.ovh.net. Модель за
    замовчуванням — gpt-oss-120b (той самий каталог має й
    Qwen3.5-397B-A17B — обидва перевірені реальні ID моделей на
    endpoints.ai.cloud.ovh.net/catalog).

    ВАЖЛИВО (перевірено наживо 2026-09-07, реальними HTTP-запитами, не за
    документацією): анонімний тір потребує, щоб заголовок Authorization не
    надсилався ВЗАГАЛІ — не порожній рядок (сервер відповідає 400 "invalid
    HTTP header"), не плейсхолдер-токен (403 "authentication failed"; саме
    так спершу й спрацювало, поки не перевірив без заголовка й отримав
    очікуваний 429 "rate limit" — підтвердження, що анонімна автентифікація
    приймається). А openai-python SDK ЗАВЖДИ додає Authorization: Bearer
    <api_key> і не дає прибрати цей заголовок повністю — тому анонімний
    шлях тут іде НАПРЯМУ через httpx (той самий контракт chat/completions,
    без пакета `openai`); сам SDK використовується лише якщо реально
    заданий OVH_AI_ENDPOINTS_ACCESS_TOKEN.

    Анонімний тір — 2 запити/хв на IP+модель (перевищення -> 429). Тому
    ТРОТТЛІНГ ОБОВ'ЯЗКОВИЙ: клас тримає час останнього виклику як
    class-level стан (спільний для всіх інстансів — усі виклики цього
    провайдера з одного процесу йдуть з одного IP, тож ліміт спільний), і
    блокуюче чекає перед кожним запитом, якщо треба. complete() —
    синхронний метод (виконується в робочому потоці через
    asyncio.to_thread, не в event loop), тому тут це time.sleep(), а не
    asyncio.sleep() — await поза корутиною неможливий; ефект той самий
    (не блокує event loop, лише цей робочий потік).
    """

    _BASE_URL = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
    _MIN_INTERVAL = 31.0  # трохи більше 60/2=30с — запас від "впритул" у ліміт
    _throttle_lock = threading.Lock()
    _last_call_at = 0.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-oss-120b",
        max_tokens: int = 4000,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._token = api_key or os.environ.get("OVH_AI_ENDPOINTS_ACCESS_TOKEN")

        if self._token:
            try:
                import openai
            except ImportError as exc:
                raise LlmSolverError(
                    "Пакет 'openai' не встановлено (pip install openai)."
                ) from exc
            self._openai = openai
            self.client = openai.OpenAI(
                api_key=self._token,
                base_url=self._BASE_URL,
                timeout=_LLM_HTTP_TIMEOUT,
                max_retries=_LLM_HTTP_MAX_RETRIES,
            )
        else:
            self._openai = None
            self.client = None
            try:
                import httpx
            except ImportError as exc:
                raise LlmSolverError(
                    "Пакет 'httpx' не встановлено (pip install httpx)."
                ) from exc
            self._httpx = httpx

    def _throttle(self) -> None:
        cls = type(self)
        with cls._throttle_lock:
            wait = cls._MIN_INTERVAL - (time.monotonic() - cls._last_call_at)
            if wait > 0:
                logger.info("OVHcloud: троттлінг — чекаю %.1fс (анонімний ліміт 2 запити/хв).", wait)
                time.sleep(wait)
            cls._last_call_at = time.monotonic()

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self._throttle()
        if self.client is not None:
            return self._complete_authenticated(system_prompt, user_prompt)
        return self._complete_anonymous(system_prompt, user_prompt)

    def _complete_authenticated(self, system_prompt: str, user_prompt: str) -> str:
        openai = self._openai
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except openai.RateLimitError as exc:
            raise RateLimitError(
                f"Денний ліміт/rate-limit вичерпано для 'ovh'. Деталі: {exc}"
            ) from exc
        except openai.AuthenticationError as exc:
            raise LlmSolverError(f"Невірний OVH_AI_ENDPOINTS_ACCESS_TOKEN: {exc}") from exc
        except openai.InternalServerError as exc:
            raise ServerUnavailableError(f"OVHcloud: серверна помилка/перевантаження: {exc}") from exc
        except openai.APIStatusError as exc:
            raise LlmSolverError(f"OVHcloud API повернув помилку: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise LlmSolverError(f"Не вдалось з'єднатись з OVHcloud API: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice and choice.message else None
        if not text:
            raise LlmSolverError("OVHcloud API повернув відповідь без тексту.")
        if choice.finish_reason == "length":
            text += self._TRUNCATED_NOTE
        return text

    _TRUNCATED_NOTE = (
        "\n\n⚠️ Відповідь обірвана лімітом довжини — модель не встигла "
        "закінчити думку. Онов max_tokens в config.yaml/OvhCloudProvider, "
        "якщо це трапляється часто."
    )

    def _complete_anonymous(self, system_prompt: str, user_prompt: str) -> str:
        httpx = self._httpx
        try:
            resp = httpx.post(
                f"{self._BASE_URL}/chat/completions",
                json={
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=60,
            )
        except httpx.HTTPError as exc:
            raise LlmSolverError(f"Не вдалось з'єднатись з OVHcloud API: {exc}") from exc

        if resp.status_code == 429:
            raise RateLimitError(
                f"Денний ліміт/rate-limit вичерпано для 'ovh' (анонімно, 2/хв). "
                f"Деталі: {resp.text[:300]}"
            )
        if resp.status_code >= 500:
            raise ServerUnavailableError(
                f"OVHcloud: серверна помилка/перевантаження {resp.status_code}: {resp.text[:300]}"
            )
        if resp.status_code >= 400:
            raise LlmSolverError(
                f"OVHcloud API повернув помилку {resp.status_code}: {resp.text[:300]}"
            )

        try:
            data = resp.json()
            choice = (data.get("choices") or [None])[0]
            text = (choice or {}).get("message", {}).get("content")
        except (ValueError, AttributeError, KeyError) as exc:
            raise LlmSolverError(f"OVHcloud API повернув нерозбірливу відповідь: {exc}") from exc

        if not text:
            raise LlmSolverError("OVHcloud API повернув відповідь без тексту.")
        if choice.get("finish_reason") == "length":
            text += self._TRUNCATED_NOTE
        return text


class ZaiProvider(LlmProvider):
    """OpenAI-сумісний SDK, base_url на api.z.ai (безкоштовна реєстрація, без картки)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "glm-4.7-flash",
        max_tokens: int = 4000,
    ):
        try:
            import openai
        except ImportError as exc:
            raise LlmSolverError(
                "Пакет 'openai' не встановлено (pip install openai)."
            ) from exc

        key = api_key or os.environ.get("ZAI_API_KEY")
        if not key:
            raise LlmSolverError(
                "Немає ZAI_API_KEY (ані в аргументі, ані в .env/оточенні)."
            )

        self._openai = openai
        self.client = openai.OpenAI(
            api_key=key,
            base_url="https://api.z.ai/api/paas/v4",
            timeout=_LLM_HTTP_TIMEOUT,
            max_retries=_LLM_HTTP_MAX_RETRIES,
        )
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        openai = self._openai
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except openai.RateLimitError as exc:
            raise RateLimitError(
                f"Денний ліміт/rate-limit вичерпано для 'zai'. Деталі: {exc}"
            ) from exc
        except openai.AuthenticationError as exc:
            raise LlmSolverError(f"Невірний ZAI_API_KEY: {exc}") from exc
        except openai.InternalServerError as exc:
            raise ServerUnavailableError(f"Z.AI: серверна помилка/перевантаження: {exc}") from exc
        except openai.APIStatusError as exc:
            raise LlmSolverError(f"Z.AI API повернув помилку: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise LlmSolverError(f"Не вдалось з'єднатись з Z.AI API: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice and choice.message else None
        if not text:
            raise LlmSolverError("Z.AI API повернув відповідь без тексту.")

        if choice.finish_reason == "length":
            logger.warning("Z.AI: відповідь обірвана лімітом max_tokens=%d.", self.max_tokens)
            text += (
                "\n\n⚠️ Відповідь обірвана лімітом довжини — модель не встигла "
                "закінчити думку. Онов max_tokens в config.yaml/ZaiProvider, "
                "якщо це трапляється часто."
            )
        return text


class CloudflareProvider(LlmProvider):
    """
    OpenAI-сумісний ендпоінт Cloudflare Workers AI — РЕЗЕРВНИЙ провайдер
    (задум користувача: "юзати як організатор якщо треба"). Підключається
    лише як ОСТАННІЙ у config.yaml -> llm.fallback_order, не як основний
    робочий вибір для жодного предмету — спрацьовує тільки коли всі інші
    провайдери в ланцюжку вже впали з rate-limit.

    base_url будується з CLOUDFLARE_ACCOUNT_ID (сам accountId не є
    секретом, але без нього URL неповний), ключ — CLOUDFLARE_API_TOKEN.
    Модель за замовчуванням — @cf/deepseek-ai/deepseek-r1-distill-qwen-32b
    (перевірено на developers.cloudflare.com/workers-ai/models — точний
    ID, з префіксом "@cf/" обов'язково).
    """

    _THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        max_tokens: int = 4000,
    ):
        try:
            import openai
        except ImportError as exc:
            raise LlmSolverError(
                "Пакет 'openai' не встановлено (pip install openai)."
            ) from exc

        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        key = api_key or os.environ.get("CLOUDFLARE_API_TOKEN")
        if not account_id or not key:
            raise LlmSolverError(
                "Немає CLOUDFLARE_ACCOUNT_ID і/чи CLOUDFLARE_API_TOKEN "
                "(ані в аргументі, ані в .env/оточенні)."
            )

        self._openai = openai
        self.client = openai.OpenAI(
            api_key=key,
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
            timeout=_LLM_HTTP_TIMEOUT,
            max_retries=_LLM_HTTP_MAX_RETRIES,
        )
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        openai = self._openai
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except openai.RateLimitError as exc:
            raise RateLimitError(
                f"Денний ліміт/rate-limit вичерпано для 'cloudflare'. Деталі: {exc}"
            ) from exc
        except openai.AuthenticationError as exc:
            raise LlmSolverError(f"Невірний CLOUDFLARE_API_TOKEN: {exc}") from exc
        except openai.InternalServerError as exc:
            raise ServerUnavailableError(f"Cloudflare: серверна помилка/перевантаження: {exc}") from exc
        except openai.APIStatusError as exc:
            raise LlmSolverError(f"Cloudflare API повернув помилку: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise LlmSolverError(f"Не вдалось з'єднатись з Cloudflare API: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice and choice.message else None
        if not text:
            raise LlmSolverError("Cloudflare API повернув відповідь без тексту.")

        # deepseek-r1-distill теж може повернути ланцюжок міркувань прямо в
        # тілі відповіді, обгорнутий у <think>...</think> — прибираємо, той
        # самий випадок, що й був у видаленого SiliconFlowProvider.
        text = self._THINK_BLOCK_RE.sub("", text).strip()
        if not text:
            raise LlmSolverError(
                "Cloudflare API повернув лише <think>-блок без фінальної відповіді."
            )

        if choice.finish_reason == "length":
            logger.warning("Cloudflare: відповідь обірвана лімітом max_tokens=%d.", self.max_tokens)
            text += (
                "\n\n⚠️ Відповідь обірвана лімітом довжини — модель не встигла "
                "закінчити думку. Онов max_tokens в config.yaml/CloudflareProvider, "
                "якщо це трапляється часто."
            )
        return text


class BrowserChatProvider(LlmProvider):
    """
    Автоматизує веб-чат (chat.deepseek.com / aistudio.google.com) через
    Playwright замість офіційного API.

    !! УВАГА: це майже напевно порушує ToS цих сервісів (автоматизований
    доступ до інтерфейсу, який призначений для ручного використання людиною)
    — реальний ризик бану акаунту, антибот-детекції, і крихкість при зміні
    верстки. Це свідомий компроміс користувача, не рекомендована конфігурація
    за замовчуванням.

    Сесія логіну зберігається в persistent Chromium-профілі (user_data_dir).
    Логін робиться один раз вручну через setup_browser_profile.py — цей клас
    сам НІКОЛИ не логіниться (паролів не зберігає, капч не розв'язує).
    """

    def __init__(
        self,
        service: str,
        model: Optional[str] = None,  # ігнорується — сумісність сигнатури з іншими провайдерами
        config: Optional[dict] = None,
    ):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise LlmSolverError(
                "Пакет 'playwright' не встановлено "
                "(pip install playwright && playwright install chromium)."
            ) from exc

        config = config if config is not None else _load_config()
        browser_cfg = config.get("browser_chat") or {}
        svc_cfg = (browser_cfg.get("services") or {}).get(service)
        if not svc_cfg:
            raise LlmSolverError(
                f"Немає config.yaml -> browser_chat.services.{service}."
            )

        try:
            self.url = svc_cfg["url"]
            self.input_selector = svc_cfg["input_selector"]
            user_data_dir = svc_cfg["user_data_dir"]
        except KeyError as exc:
            raise LlmSolverError(
                f"config.yaml -> browser_chat.services.{service} не вистачає "
                f"обов'язкового поля {exc}."
            ) from exc

        self.service = service
        self.user_data_dir = str(Path(user_data_dir).expanduser())
        self.submit_key = svc_cfg.get("submit_key", "Enter")
        self.response_container_selector = svc_cfg.get("response_container_selector")
        self.login_form_selector = svc_cfg.get("login_form_selector")
        self.login_url_pattern = svc_cfg.get("login_url_pattern")

        self.wait_timeout = browser_cfg.get("wait_timeout", 60)
        self.poll_interval = browser_cfg.get("poll_interval", 1.5)
        self.stable_checks = browser_cfg.get("stable_checks", 2)
        self.input_ready_timeout = browser_cfg.get("input_ready_timeout", 20)
        self.input_ready_retries = browser_cfg.get("input_ready_retries", 3)
        self.headless = browser_cfg.get("headless", False)
        self.channel = browser_cfg.get("channel")  # напр. "chrome" — реальний встановлений Google Chrome замість Playwright-Chromium

        if self.headless:
            logger.warning(
                "browser_chat.headless=true для '%s' — headless-режим частіше "
                "ловить антибот-блокування (капчі, Cloudflare) ніж звичайне вікно.",
                service,
            )

        Path(self.user_data_dir).mkdir(parents=True, exist_ok=True)

        launch_kwargs: dict = {"headless": self.headless}
        if self.channel:
            launch_kwargs["channel"] = self.channel

        self._playwright = sync_playwright().start()
        try:
            self._context = self._playwright.chromium.launch_persistent_context(
                self.user_data_dir, **launch_kwargs
            )
        except Exception as exc:
            self._playwright.stop()
            raise LlmSolverError(
                f"Не вдалось запустити браузер для '{service}' "
                f"(channel={self.channel or 'bundled chromium'}): {exc}"
            ) from exc
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()

    def close(self) -> None:
        try:
            self._context.close()
        finally:
            self._playwright.stop()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _check_session(self) -> None:
        page = self._page
        expired = bool(self.login_url_pattern and self.login_url_pattern in page.url)

        if not expired and self.login_form_selector:
            try:
                expired = page.locator(self.login_form_selector).first.is_visible(timeout=3000)
            except Exception:
                expired = False

        if expired:
            raise BrowserSessionExpiredError(
                f"Сесія профілю '{self.service}' злетіла (бачу форму логіну "
                f"замість чату за адресою {page.url}). Онови її вручну:\n"
                f"  python setup_browser_profile.py --service {self.service}\n"
                f"Профіль: {self.user_data_dir}"
            )

    def _wait_for_input(self):
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.input_ready_retries + 1):
            try:
                locator = self._page.locator(self.input_selector).first
                locator.wait_for(state="visible", timeout=self.input_ready_timeout * 1000)
                return locator
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Поле вводу '%s' на '%s' не з'явилось (спроба %d/%d): %s",
                    self.input_selector, self.service, attempt, self.input_ready_retries, exc,
                )
        raise LlmSolverError(
            f"Поле вводу '{self.input_selector}' на '{self.service}' так і не "
            f"з'явилось за {self.input_ready_retries} спроб — можливо, "
            f"input_selector в config.yaml застарів."
        ) from last_exc

    def _snapshot_text(self) -> str:
        try:
            if self.response_container_selector:
                return self._page.locator(self.response_container_selector).last.inner_text(
                    timeout=2000
                )
            return self._page.locator("body").inner_text(timeout=2000)
        except Exception:
            return ""

    def _wait_for_response(self, baseline: str) -> str:
        deadline = time.monotonic() + self.wait_timeout
        previous: Optional[str] = None
        stable_count = 0

        while time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            current = self._snapshot_text()

            if current and current == previous and current != baseline:
                stable_count += 1
                if stable_count >= self.stable_checks:
                    return self._extract_answer(current, baseline)
            else:
                stable_count = 0
            previous = current

        logger.warning(
            "Таймаут очікування відповіді (%ss) на '%s' — повертаю останній знімок тексту.",
            self.wait_timeout,
            self.service,
        )
        return self._extract_answer(previous or "", baseline)

    def _extract_answer(self, current: str, baseline: str) -> str:
        if self.response_container_selector:
            # Селектор вже вказує саме на блок відповіді - віднімати нічого не треба.
            return current.strip()
        # Немає надійного селектора (типово для DeepSeek — хешовані CSS-класи,
        # що змінюються з кожним білдом): фолбек — різниця тексту всієї
        # сторінки до і після відправки повідомлення.
        if current.startswith(baseline):
            return current[len(baseline):].strip()
        return current.strip()

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        page = self._page
        response = page.goto(self.url, wait_until="domcontentloaded")
        if response is not None and response.status >= 400:
            raise LlmSolverError(
                f"'{self.service}' повернув HTTP {response.status} при заході на "
                f"{self.url} — це схоже на антибот-блокування (CloudFront/WAF), "
                f"а не на протухлу сесію. Особливо ймовірно з дата-центрового IP "
                f"(сервер/хмара) — саме той ризик, про який попереджали для "
                f"BrowserChatProvider. Спробуй з іншої мережі, або повернись на "
                f"офіційний API-провайдер."
            )
        self._check_session()

        input_el = self._wait_for_input()
        message = f"{system_prompt}\n\n---\n\n{user_prompt}"

        baseline = self._snapshot_text()

        input_el.click()
        input_el.fill(message)
        page.keyboard.press(self.submit_key)

        answer = self._wait_for_response(baseline)
        if not answer:
            raise LlmSolverError(
                f"Не вдалось витягнути відповідь з чату '{self.service}' — "
                f"можливо, розмітка сайту змінилась (перевір "
                f"response_container_selector в config.yaml)."
            )
        return answer


_PROVIDER_CLASSES: dict[str, Callable[..., LlmProvider]] = {
    "groq": lambda model=None, config=None: GroqProvider(
        **({"model": model} if model else {})
    ),
    "gemini": lambda model=None, config=None: GeminiProvider(
        **({"model": model} if model else {})
    ),
    "ovh": lambda model=None, config=None: OvhCloudProvider(
        **({"model": model} if model else {})
    ),
    "zai": lambda model=None, config=None: ZaiProvider(
        **({"model": model} if model else {})
    ),
    "cloudflare": lambda model=None, config=None: CloudflareProvider(
        **({"model": model} if model else {})
    ),
    "deepseek_browser": lambda model=None, config=None: BrowserChatProvider(
        service="deepseek", config=config
    ),
    "gemini_browser": lambda model=None, config=None: BrowserChatProvider(
        service="gemini", config=config
    ),
}


def _build_user_prompt(task: Task) -> str:
    parts = [f"Предмет: {task.subject}", f"Завдання: {task.exercise_source_text}"]
    if task.book_page and (task.book_page.get("page") or task.book_page.get("exercise")):
        ref = []
        if task.book_page.get("page"):
            ref.append(f"сторінка {task.book_page['page']}")
        if task.book_page.get("exercise"):
            ref.append(f"вправа/номер {task.book_page['exercise']}")
        parts.append("Довідково з тексту ДЗ: " + ", ".join(ref))
    return "\n".join(parts)


class LlmSolver:
    """
    Обирає LLM-провайдера (і модель) за предметом завдання, за правилами
    з config.yaml -> llm (default_provider / overrides / providers).
    Провайдери кешуються за (ім'я, модель), щоб не переавторизовуватись
    на кожен виклик.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config if config is not None else _load_config()
        self._provider_cache: dict[tuple[str, Optional[str]], LlmProvider] = {}
        # telegram_bot.py тепер розв'язує декілька Task паралельно (окремі
        # робочі потоки через asyncio.to_thread) — без локу конкурентний
        # перший виклик _get_provider() для того самого (name, model) міг би
        # створити провайдера двічі (check-then-act без атомарності).
        self._provider_lock = threading.Lock()

    def _resolve_provider_and_model(self, subject: str) -> tuple[str, Optional[str]]:
        llm_cfg = self.config.get("llm") or {}
        overrides = llm_cfg.get("overrides") or {}
        provider_name = overrides.get(subject) or llm_cfg.get("default_provider", "groq")
        return provider_name, self._model_for_provider(provider_name, subject)

    def _model_for_provider(self, provider_name: str, subject: str) -> Optional[str]:
        llm_cfg = self.config.get("llm") or {}
        provider_cfg = (llm_cfg.get("providers") or {}).get(provider_name) or {}
        model_overrides = provider_cfg.get("model_overrides") or {}
        return model_overrides.get(subject) or provider_cfg.get("default_model")

    def _fallback_chain(self, provider_name: str) -> list[str]:
        """
        Ланцюжок провайдерів для одного solve()-виклику: спершу обраний за
        правилами (default_provider/overrides), далі — config.yaml ->
        llm.fallback_order, без дублікатів. Якщо fallback_order не задано —
        ланцюжок з одного елемента (стара поведінка, без failover'у).
        """
        llm_cfg = self.config.get("llm") or {}
        fallback_order = llm_cfg.get("fallback_order") or []
        chain = [provider_name]
        for name in fallback_order:
            if name not in chain:
                chain.append(name)
        return chain

    def _get_provider(self, name: str, model: Optional[str]) -> LlmProvider:
        cache_key = (name, model)
        if cache_key not in self._provider_cache:
            with self._provider_lock:
                # Подвійна перевірка під локом — інший потік міг встигнути
                # створити провайдера, поки ми чекали на лок.
                if cache_key not in self._provider_cache:
                    factory = _PROVIDER_CLASSES.get(name)
                    if factory is None:
                        raise LlmSolverError(
                            f"Невідомий провайдер '{name}' в config.yaml "
                            f"(llm.default_provider/overrides) — доступні: "
                            f"{', '.join(_PROVIDER_CLASSES)}."
                        )
                    self._provider_cache[cache_key] = factory(model=model, config=self.config)
        return self._provider_cache[cache_key]

    def solve(
        self,
        task: Task,
        mode: Literal["answer", "explain", "write"] = "explain",
        assessment_prep: bool = False,
    ) -> str:
        """
        Пробує провайдерів по черзі за self._fallback_chain(): якщо провайдер
        падає з RateLimitError (429/RESOURCE_EXHAUSTED), пробує наступного в
        ланцюжку замість того, щоб одразу здатись. Інші помилки (авторизація,
        мережа, невідома конфігурація) не ловляться тут — вони пролітають
        одразу вгору, бо це майже завжди проблема конфігурації, яку тихий
        fallback лише замаскував би.

        mode="write" — окремий режим для write_task (solver.py ->
        classify_task): LLM пише ГОТОВИЙ текст твору/есе, а не
        пояснення/план. assessment_prep стосується лише mode="explain" —
        для "write" ігнорується (написання твору й підготовка до
        контрольної — взаємовиключні сценарії).

        Fix #2: для mode="explain" використовує предметно-адаптивні системні
        промти: _EXPLAIN_SYSTEM_EXACT (математика/фізика/хімія) або
        _EXPLAIN_SYSTEM_HUMANITIES (інші предмети).

        Якщо ВСІ провайдери в ланцюжку впали з rate-limit — кидає
        LlmSolverError з коротким людським повідомленням (повні деталі
        останньої помилки йдуть лише в лог, не в текст винятку, щоб їх не
        побачив користувач у Telegram).
        """
        if mode == "write":
            system = _WRITE_SYSTEM
        elif mode == "explain":
            subject = task.subject
            is_exact_science = (
                "Математика" in subject
                or "Фізика" in subject
                or "Хімія" in subject
                or "Біологія" in subject
            )
            system = (
                _EXPLAIN_SYSTEM_EXACT
                if is_exact_science
                else _EXPLAIN_SYSTEM_HUMANITIES
            )
            if assessment_prep:
                system += _ASSESSMENT_PREP_ADDENDUM
        else:
            system = _ANSWER_SYSTEM
        user_content = _build_user_prompt(task)

        provider_name, _ = self._resolve_provider_and_model(task.subject)
        chain = self._fallback_chain(provider_name)

        last_exc: Optional[Exception] = None
        for idx, name in enumerate(chain):
            model = self._model_for_provider(name, task.subject)
            try:
                provider = self._get_provider(name, model)
            except LlmSolverError as exc:
                logger.warning("Провайдер '%s' недоступний (ініціалізація): %s", name, exc)
                last_exc = exc
                continue

            logger.info(
                "Розв'язую '%s' через провайдера '%s' (модель: %s)%s.",
                task.subject,
                name,
                model or "(дефолтна для провайдера)",
                "" if idx == 0 else " [fallback]",
            )
            try:
                return _call_provider(
                    provider, system, user_content, self.config,
                    subject=task.subject, provider_name=name,
                )
            except TransientProviderError as exc:
                reason = "rate-limit/quota" if isinstance(exc, RateLimitError) else "серверна помилка"
                logger.warning(
                    "Провайдер '%s' впав (%s): %s — пробую наступного в ланцюжку %s.",
                    name,
                    reason,
                    exc,
                    chain[idx + 1:] or "(більше нема)",
                )
                last_exc = exc
                continue

        logger.error(
            "Усі провайдери в ланцюжку %s впали для '%s' ('%s'). Остання помилка: %r",
            chain,
            task.subject,
            task.homework_text,
            last_exc,
        )
        raise LlmSolverError(
            "⚠️ Не вдалось розв'язати (усі провайдери недоступні), спробуй пізніше."
        ) from last_exc


# ---------------------------------------------------------------------- #
# OCR-кеш (SQLite) і звірка скану ГДЗ з LLM-відповіддю
# ---------------------------------------------------------------------- #

def _ocr_db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS gdz_ocr_cache ("
        "image_url TEXT, kind TEXT NOT NULL DEFAULT 'answer', "
        "ocr_text TEXT NOT NULL, timestamp TEXT NOT NULL, "
        "PRIMARY KEY (image_url, kind))"
    )
    # М'яка міграція зі старої схеми (image_url TEXT PRIMARY KEY, без kind) —
    # той самий скан тепер кешується окремо для "умова" і "відповідь", бо
    # це різні промти й різний текст на виході.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(gdz_ocr_cache)").fetchall()}
    if "kind" not in cols:
        conn.execute("ALTER TABLE gdz_ocr_cache RENAME TO gdz_ocr_cache_v1")
        conn.execute(
            "CREATE TABLE gdz_ocr_cache ("
            "image_url TEXT, kind TEXT NOT NULL DEFAULT 'answer', "
            "ocr_text TEXT NOT NULL, timestamp TEXT NOT NULL, "
            "PRIMARY KEY (image_url, kind))"
        )
        conn.execute(
            "INSERT INTO gdz_ocr_cache (image_url, kind, ocr_text, timestamp) "
            "SELECT image_url, 'answer', ocr_text, timestamp FROM gdz_ocr_cache_v1"
        )
        conn.execute("DROP TABLE gdz_ocr_cache_v1")
        conn.commit()
    return conn


def _ocr_cache_get(image_url: str, kind: str = "answer") -> Optional[str]:
    conn = _ocr_db_connect()
    try:
        row = conn.execute(
            "SELECT ocr_text FROM gdz_ocr_cache WHERE image_url = ? AND kind = ?",
            (image_url, kind),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _ocr_cache_set(image_url: str, ocr_text: str, kind: str = "answer") -> None:
    conn = _ocr_db_connect()
    try:
        conn.execute(
            "INSERT INTO gdz_ocr_cache (image_url, kind, ocr_text, timestamp) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(image_url, kind) DO UPDATE SET "
            "ocr_text=excluded.ocr_text, timestamp=excluded.timestamp",
            (image_url, kind, ocr_text, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


_OCR_ANSWER_PROMPT = (
    "Розпізнай текст/формули з цього зображення розв'язку задачі. "
    "Поверни точний текст без пояснень."
)

_OCR_CONDITION_PROMPT = (
    "На цьому зображенні — сторінка з ГДЗ-розв'язника або підручника. "
    "Розпізнай ЛИШЕ УМОВУ задачі (постановку завдання), без розв'язку і без "
    "відповіді. Якщо на зображенні є і умова, і розв'язок разом (типово для "
    "ГДЗ-сканів) — витягни тільки частину з умовою, зупинись, щойно "
    "починається розв'язання/відповідь. Якщо умови на зображенні взагалі "
    "немає (є лише номер і розв'язок, без переозвучення завдання) — поверни "
    "рівно текст: NO_CONDITION_FOUND. Нічого іншого не додавай."
)

_NO_CONDITION_MARKER = "NO_CONDITION_FOUND"


def _ocr_scan(
    image_url: str,
    api_key: Optional[str] = None,
    prompt: str = _OCR_ANSWER_PROMPT,
    kind: str = "answer",
) -> str:
    """
    OCR скану через Gemini Vision (gemini-2.5-flash), з кешем у SQLite
    (nz_solver.db -> gdz_ocr_cache), щоб не платити повторний Vision-виклик
    на той самий (image_url, kind). kind розділяє кеш для різних промтів
    на тому самому зображенні (напр. "answer" vs "condition") — інакше
    результат одного промта підмінив би результат іншого.
    """
    t0 = time.monotonic()
    cached = _ocr_cache_get(image_url, kind=kind)
    if cached is not None:
        logger.info("OCR (%s): знайдено в кеші для %s.", kind, image_url)
        timing_logger.info(
            "[timing] ocr_scan_cache_hit kind=%s image=%s: %.3fs", kind, image_url, time.monotonic() - t0
        )
        return cached

    try:
        from google import genai
        from google.genai import errors as genai_errors
        from google.genai import types as genai_types
    except ImportError as exc:
        raise LlmSolverError(
            "Пакет 'google-genai' не встановлено (pip install google-genai)."
        ) from exc

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise LlmSolverError(
            "Немає GEMINI_API_KEY для OCR (ані в аргументі, ані в .env/оточенні)."
        )

    import requests

    try:
        img_resp = requests.get(image_url, timeout=15)
        img_resp.raise_for_status()
    except requests.RequestException as exc:
        raise LlmSolverError(f"Не вдалось завантажити зображення {image_url}: {exc}") from exc

    mime_type = img_resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()

    client = genai.Client(
        api_key=key, http_options=genai_types.HttpOptions(timeout=int(_LLM_HTTP_TIMEOUT * 1000))
    )
    t_vision0 = time.monotonic()
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                genai_types.Part.from_bytes(data=img_resp.content, mime_type=mime_type),
                prompt,
            ],
        )
    except genai_errors.ClientError as exc:
        raise LlmSolverError(f"OCR: Gemini Vision — помилка клієнта: {exc}") from exc
    except genai_errors.ServerError as exc:
        raise LlmSolverError(f"OCR: Gemini Vision — серверна помилка: {exc}") from exc
    except genai_errors.APIError as exc:
        raise LlmSolverError(f"OCR: Gemini Vision повернув помилку: {exc}") from exc
    finally:
        timing_logger.info(
            "[timing] ocr_scan_vision_call kind=%s image=%s: %.3fs",
            kind, image_url, time.monotonic() - t_vision0,
        )

    text = getattr(response, "text", None)
    if not text:
        raise LlmSolverError("OCR: Gemini Vision повернув відповідь без тексту.")

    _ocr_cache_set(image_url, text, kind=kind)
    return text


_COMPARE_SYSTEM = (
    "Ти звіряєш дві відповіді на одну шкільну задачу. Відповідай ЛИШЕ одним "
    "словом: 'так' — якщо відповіді збігаються по суті (той самий числовий "
    "результат чи ключовий висновок, стиль/пояснення не важливі), або 'ні' "
    "— якщо результати різні. Без жодних інших слів."
)


def _answers_match(llm: LlmSolver, task: Task, answer_a: str, answer_b: str) -> bool:
    provider_name, model = llm._resolve_provider_and_model(task.subject)
    provider = llm._get_provider(provider_name, model)
    prompt = (
        f"Задача: {task.exercise_source_text}\n\n"
        f"Відповідь A:\n{answer_a}\n\nВідповідь B:\n{answer_b}"
    )
    verdict = _call_provider(
        provider, _COMPARE_SYSTEM, prompt, llm.config,
        subject=task.subject, provider_name=provider_name,
    )
    return verdict.strip().lower().startswith("так")


# ---------------------------------------------------------------------- #
# solve_task
# ---------------------------------------------------------------------- #

def _condition_from_gdz_scan(image_url: str, api_key: Optional[str] = None) -> Optional[str]:
    """
    OCR-нути УМОВУ задачі зі скану ГДЗ (не відповідь, а саме постановка завдання).
    Якщо умоди немає — повертає None (або NO_CONDITION_MARKER).
    """
    try:
        logger.info("OCR умови зі скану ГДЗ: %s", image_url)
        condition_text = _ocr_scan(image_url, prompt=_OCR_CONDITION_PROMPT, kind="condition")
        logger.info("OCR-умова отримана: %s", condition_text[:200] if len(condition_text) > 200 else condition_text)
        if condition_text == _NO_CONDITION_MARKER or not condition_text.strip():
            logger.info("OCR умови не знайдено (NO_CONDITION_MARKER або порожній текст)")
            return None
        logger.info("OCR умова успішно отримана")
        return condition_text
    except LlmSolverError as exc:
        logger.warning("OCR умови не вдався для %s: %s", image_url, exc)
        return None


def _is_non_core_subject(subject: str) -> bool:
    """
    Перевіряє, чи є предмет непрофільним (не математика, фізика, хімія, біологія).
    Для непрофільних предметів не треба робити сувений sanity-check через OCR.
    """
    return not any(
        kw in subject
        for kw in ("Математика", "Фізика", "Хімія", "Біологія")
    )


def _condition_matches(task_condition: str, gdz_condition: str, llm: LlmSolver, task: Task) -> bool:
    """
    Перевіряє, чи є дві умови задачею одну й ту саму задачу.
    Використовує LLM для семантичного порівняння.
    
    Для непрофільних предметів (українська мова, історія тощо) це занадто суворо,
    тому повертає True без перевірки — GDZ можна використовувати.
    """
    if _is_non_core_subject(task.subject):
        logger.debug("Непрофільний предмет '%s' — пропускаю сувений sanity-check", task.subject)
        return True
    
    provider_name, model = llm._resolve_provider_and_model(task.subject)
    provider = llm._get_provider(provider_name, model)
    
    # DEBUG: логування перед порівнянням
    logger.info(
        "=== SANITY-CHECK DEBUG ===\n"
        "Ціль (task.exercise_source_text):\n%s\n\n"
        "Знайдено в ГДЗ (OCR-умова зі скану):\n%s\n\n"
        "Порівнюю через LLM...",
        task_condition, gdz_condition
    )
    
    prompt = (
        f"Чи це та сама шкільна задача?\n\n"
        f"Варіант A (що шукали):\n{task_condition}\n\n"
        f"Варіант B (зі скану ГДЗ):\n{gdz_condition}\n\n"
        f"Відповідь ЛИШЕ 'так' або 'ні'."
    )
    
    # DEBUG: вивести повний промпт для аналізу
    logger.info(
        "=== SANITY-CHECK PROMPT ===\n"
        "System prompt:\n%s\n\n"
        "User prompt:\n%s",
        _COMPARE_SYSTEM, prompt
    )
    
    try:
        verdict = _call_provider(
            provider, _COMPARE_SYSTEM, prompt, llm.config,
            subject=task.subject, provider_name=provider_name,
        )
        logger.info("LLM відповів на порівняння: %s", verdict.strip())
        result = verdict.strip().lower().startswith("так")
        logger.info("=== SANITY-CHECK RESULT: %s ===", "MATCH" if result else "NO MATCH")
        return result
    except LlmSolverError as exc:
        logger.warning("Порівняння умов не вдалось: %s — вважаю розбіжністю.", exc)
        return False


def _is_garbage_gdz_text(text: str) -> bool:
    """
    Перевіряє, чи GDZ-текст є "мусором" (головна сторінка, список підручників тощо).
    Такі скани не містять конкретної відповіді на вправу.
    """
    garbage_patterns = [
        r"^[Пп]ідручники\s*\d+\s*клас.*$",
        r"^[Гг]ДЗ\s*\d+\s*клас.*$",
        r"[Пп]ідручники.*\d+\s*клас.*[Гг]ДЗ",
        r"[Дд][Пп][Аа]\s*\d+\s*клас.*$",
        r"^[Зз][Нн][Оо]$",
        r"^[Рр]еферати$",
        r"^[Аа]втори$",
        r"^[Кк]онтакти$",
        r"^[Пп]олітика\s*[Кк]онфідентності$",
        r"^[Мм]атеріали\s*[Сс]айту.*мають\s*[Аа]вторське\s*[Пп]раво.*$",
        r"^[Рр]озміщення\s*[Бб]удь-якої\s*[Іі]нформації.*порушує\s*[Аа]вторське\s*[Пп]раво.*$",
    ]
    import re
    text_lower = text.lower()
    for pat in garbage_patterns:
        if re.search(pat, text_lower, re.IGNORECASE):
            return True
    
    # Якщо текст занадто довгий (більше 5000 символів), це може бути головна сторінка з меню
    if len(text) > 5000:
        return True
    
    # Якщо текст занадто короткий (менше 20 символів), це може бути помилка OCR
    if len(text) < 20:
        return True
    
    # Якщо текст містить багато "менюшних" слів, це може бути головна сторінка
    menu_words = ["підручники", "гдз", "дпа", "зно", "реферати", "автори", "контакти", "політика", "конфідентності"]
    menu_word_count = sum(1 for word in menu_words if word in text_lower)
    if menu_word_count >= 3:
        return True
    
    return False


def _compare_with_gdz_scan(
    task: Task, image_url: str, llm: LlmSolver, llm_answer: str, base_source: str
) -> dict:
    """
    llm_answer — ВЖЕ готова LLM-відповідь (порахована ПАРАЛЕЛЬНО з пошуком
    цього скану в solve_task, а не після нього). Тут лишається OCR-нути
    саму відповідь зі скану й звірити з llm_answer. base_source —
    джерело умови без звірки ("pdf+llm"/"llm"), стає "gdz+llm", якщо
    звірка відбулась (незалежно від збігу/розбіжності — сам факт звірки
    підвищує/знижує довіру).
    
    Fix #1: перед звіркою відповідей перевіряємо, чи скан насправді містить
    ту саму умову, що й в завданні. Якщо ні — відкидаємо GDZ повністю.
    """
    try:
        gdz_text = _ocr_scan(image_url)
    except LlmSolverError as exc:
        logger.warning(
            "OCR відповіді не вдався для %s: %s — повертаю LLM-відповідь без звірки.",
            image_url,
            exc,
        )
        return {"source": base_source, "answer": llm_answer, "confidence": "low"}
    
    # Перевірка на "мусорні" скани (головна сторінка ГДЗ, список підручників тощо)
    if _is_garbage_gdz_text(gdz_text):
        logger.warning(
            "GDZ-текст є мусором (головна сторінка/список підручників): %s... — відкидаю GDZ.",
            gdz_text[:100] if len(gdz_text) > 100 else gdz_text
        )
        return {"source": base_source, "answer": llm_answer, "confidence": "low"}

    gdz_condition = _condition_from_gdz_scan(image_url)
    if gdz_condition is not None:
        task_condition = task.exercise_source_text
        if not _condition_matches(task_condition, gdz_condition, llm, task):
            logger.info(
                "OCR-умова зі скану не збігається з завданням — відкидаю GDZ. "
                "task=%s, scan_condition=%s",
                task_condition[:100],
                gdz_condition[:100],
            )
            return {"source": base_source, "answer": llm_answer, "confidence": "low"}

    # Для нетематичних предметів (українська, історія тощо) не треба викликати _answers_match(),
    # бо OCR+LLM порівняння завжди буде помилковим (OCR читає форматування неправильно).
    # Якщо умова співпала (вище), то вважаємо відповідь правильною.
    if _is_non_core_subject(task.subject):
        logger.debug("Непрофільний предмет '%s' — пропускаю звірку відповідей LLM<->ГДЗ", task.subject)
        return {
            "source": "gdz+llm",
            "answer": f"{llm_answer}\n\n✅ Звірено з ГДЗ (непрофільний предмет — звірка відповідей пропущена).",
            "confidence": "high",
        }

    try:
        matches = _answers_match(llm, task, llm_answer, gdz_text)
    except LlmSolverError as exc:
        logger.warning("Звірка LLM<->ГДЗ не вдалась: %s — вважаю розбіжністю.", exc)
        matches = False

    if matches:
        return {
            "source": "gdz+llm",
            "answer": f"{llm_answer}\n\n✅ Звірено з ГДЗ — відповіді збігаються.",
            "confidence": "high",
        }

    return {
        "source": "gdz+llm",
        "answer": (
            "⚠️ Розбіжність між LLM і ГДЗ — потрібна перевірка людиною.\n\n"
            f"LLM-пояснення:\n{llm_answer}\n\n"
            f"Відповідь з ГДЗ (розпізнано OCR):\n{gdz_text}"
        ),
        "confidence": "low",
    }


def _condition_from_textbook_pdf(task: Task, config: dict) -> Optional[str]:
    """
    Пробує витягти реальну умову задачі (чи текст параграфа) з PDF
    підручника (config.yaml -> textbooks[subject]). None, якщо підручник
    не вказано, немає ні номера вправи, ні параграфа, чи текст не
    знайшовся — ніколи не кидає виняток (мережа/PDF — best-effort primary
    джерело, solve_task завжди має чим фолбекнутись).

    Три гілки, за тим, що знайшлось у ДЗ (nz_client.extract_book_page),
    перевіряються в цьому порядку (перший знайдений текст — переможець):
    - "exercise" (є конкретна вправа, напр. №27.5) -> extract_exercise_condition,
      умова цієї вправи.
    - "paragraph" (лише "Опрацювати §N"/"Прочитати параграф N", без номера
      вправи) -> extract_paragraph_text, весь текст параграфа для читання.
      ВАЖЛИВО: раніше цей випадок узагалі не оброблявся — book_page.get
      ("exercise") був None навіть коли текст ДЗ явно посилався на §N, тому
      PDF ніколи не підключався, хоча textbooks[subject] міг бути заданий
      (бот тоді писав "не маю тексту параграфа", хоча джерело було під
      рукою — баг знайдений і виправлений 2026-09-07).
    - лише "page" (жодного §N/номера вправи, напр. "Опрацювати матеріал на
      ст.3-10, скласти план") -> extract_page_range_text, текст указаних
      сторінок. Той самий клас бага, що й попередній — не оброблялось
      узагалі, LLM завжди йшов без реального тексту твору/розділу (баг
      знайдений і виправлений 2026-09-09).
    """
    if not task.book_page:
        return None

    exercise = task.book_page.get("exercise")
    paragraph = task.book_page.get("paragraph")
    page_only = task.book_page.get("page")
    if not exercise and not paragraph and not page_only:
        return None

    textbook_url = _lookup_by_subject(config.get("textbooks"), task.subject)
    if not textbook_url:
        return None

    cache_dir = config.get("textbook_cache_dir") or str(DEFAULT_CACHE_DIR)
    try:
        pdf_path = download_textbook(textbook_url, cache_dir)
    except Exception as exc:
        logger.info("Не вдалось завантажити підручник %s: %s", textbook_url, exc)
        return None

    if exercise:
        try:
            with _timed("pdf_extract_exercise", subject=task.subject, number=exercise):
                if _subject_may_have_figures(task.subject):
                    condition = extract_exercise_condition_with_figure(
                        pdf_path, exercise, task.book_page.get("page")
                    )
                else:
                    # Vision-опис малюнка — лише для предметів, де рисунки
                    # взагалі бувають (геометрія/математика/фізика). Для
                    # решти (укр. мова, історія тощо) виклик Vision на
                    # кожну вправу був чистими витратами: реального
                    # малюнка нема, а модель однаково повертала опис на
                    # кшталт "На зображенні відсутні геометричні фігури…",
                    # який потім приліплювався до умови й плутав LLM —
                    # реальний баг, знайдений і виправлений 2026-09-09.
                    condition = extract_exercise_condition(
                        pdf_path, exercise, task.book_page.get("page")
                    )
            if condition:
                return condition
        except Exception as exc:
            logger.info("Не вдалось витягти умову вправи з %s: %s", pdf_path, exc)

    if paragraph:
        try:
            with _timed("pdf_extract_paragraph", subject=task.subject, number=paragraph):
                condition = extract_paragraph_text(pdf_path, paragraph)
            if condition:
                return condition
        except Exception as exc:
            logger.info("Не вдалось витягти текст параграфа з %s: %s", pdf_path, exc)

    page = task.book_page.get("page")
    if page and not exercise and not paragraph:
        # Лише діапазон сторінок, без §N/номера вправи — типово для ДЗ на
        # кшталт "Опрацювати матеріал на ст.3-10, скласти план" (читання
        # суцільного тексту твору/розділу). Раніше цей випадок взагалі не
        # мав шляху вилучення умови — реальний баг, знайдений і виправлений
        # 2026-09-09.
        try:
            with _timed("pdf_extract_page_range", subject=task.subject, number=page):
                return extract_page_range_text(pdf_path, page, task.book_page.get("page_end"))
        except Exception as exc:
            logger.info("Не вдалось витягти текст сторінок з %s: %s", pdf_path, exc)

    return None


def _solve_multi_exercise(
    task: Task,
    numbers: list[str],
    mode: Literal["answer", "explain"],
    gdz: Optional[GdzSource],
    llm: Optional[LlmSolver],
    config: dict,
) -> dict:
    """
    ДЗ згадує КІЛЬКА номерів вправ через кому/крапку (напр. "розв'язати
    №27.10. 27.12, 27.13, 27.16") — раніше вся ця вправа йшла в LLM/ГДЗ як
    ОДНЕ завдання: LLM бачив лише перший номер (extract_book_page бере
    тільки його), а PDF-умова/ГДЗ-звірка для решти номерів або взагалі не
    відбувалась, або (при пошуку в ГДЗ за діапазоном сторінки) змішувала
    відповіді сусідніх номерів в одному сканованому зображенні. Тепер
    кожен номер розв'язується й звіряється ОКРЕМО (рекурсивний виклик
    solve_task з book_page, де exercise = саме цей номер) — реальний
    кейс і баг знайдені 2026-09-08.

    ВАЖЛИВО (промт 44): номери розв'язуються ПАРАЛЕЛЬНО (ThreadPoolExecutor),
    а не один за одним у циклі — послідовний for-цикл тут множив би час
    /week лінійно на кількість номерів (той самий баг, що вже колись
    виправляли для предметів між собою — тут та сама проблема всередині
    ОДНОГО предмета). Загальна кількість одночасних мережевих LLM-викликів
    все одно обмежена _llm_semaphore (config.yaml -> llm.max_concurrent_calls),
    тож розпаралелення тут не б'є по rate-limit сильніше, ніж уже й так
    б'ють паралельні предмети/дні з telegram_bot.py.
    """
    gdz = gdz or Book4Source(config=config)
    llm = llm or LlmSolver(config=config)

    def _solve_one(number: str) -> dict:
        sub_book_page = dict(task.book_page or {})
        sub_book_page["exercise"] = number
        sub_task = Task(
            subject=task.subject,
            homework_text=task.homework_text,
            book_page=sub_book_page,
            exercise_source_text=task.homework_text,
        )
        return solve_task(
            sub_task, mode=mode, gdz=gdz, llm=llm, config=config, _skip_multi_split=True
        )

    with _timed("solve_multi_exercise", subject=task.subject, count=len(numbers)):
        # Реальні ДЗ рідко згадують більш ніж 5-10 номерів — cap тут лише
        # захист від виродженого вводу (номерів десятки), щоб не плодити
        # стільки ж ОС-потоків одразу; фактична одночасність мережевих
        # LLM-викликів все одно обмежена _llm_semaphore окремо.
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(numbers), 10)) as executor:
            # executor.map зберігає порядок РЕЗУЛЬТАТІВ = порядку numbers,
            # незалежно від того, який номер порахувався першим.
            results = list(executor.map(_solve_one, numbers))

    parts = []
    sources = []
    confidences = []
    for number, result in zip(numbers, results):
        parts.append(f"**№{number}:**\n{result['answer']}")
        sources.append(result["source"])
        confidences.append(result["confidence"])

    return {
        "source": "+".join(dict.fromkeys(sources)),
        "answer": "\n\n".join(parts),
        "confidence": "high" if all(c == "high" for c in confidences) else "low",
    }


def solve_task(
    task: Task,
    mode: Literal["answer", "explain"] = "explain",
    gdz: Optional[GdzSource] = None,
    llm: Optional[LlmSolver] = None,
    config: Optional[dict] = None,
    _skip_multi_split: bool = False,
) -> dict:
    """
    Повертає {"source": "pdf+llm"|"gdz"|"gdz+llm"|"llm"|"skipped", "answer": str,
    "confidence": "high"|"low"}. Джерело може бути й комбінацією через "+"
    (напр. "pdf+llm+gdz+llm"), якщо ДЗ містило кілька номерів вправ і
    _solve_multi_exercise об'єднала результати з різними джерелами.

    Тонка обгортка навколо _solve_task_impl лише заради timing-логу
    "solve_task_total" (промт 44) — для одного номера й для КОЖНОГО
    під-номера з _solve_multi_exercise окремо (виклики рекурсивні, тож
    вкладені timing-записи тут очікувані й корисні: видно і загальний час
    предмета, і час кожного номера всередині нього).
    """
    config = config if config is not None else _load_config()
    exercise_for_log = (task.book_page or {}).get("exercise")
    with _timed("solve_task_total", subject=task.subject, exercise=exercise_for_log):
        return _solve_task_impl(task, mode, gdz, llm, config, _skip_multi_split)


def _solve_task_impl(
    task: Task,
    mode: Literal["answer", "explain"],
    gdz: Optional[GdzSource],
    llm: Optional[LlmSolver],
    config: dict,
    _skip_multi_split: bool,
) -> dict:
    if is_review_only(task.homework_text, config=config):
        return {
            "source": "skipped",
            "answer": "Просто повторити конспект/підручник — детальний розбір не потрібен.",
            "confidence": "high",
        }

    if not _skip_multi_split:
        numbers = extract_all_exercises(task.homework_text)
        if len(numbers) > 1:
            return _solve_multi_exercise(task, numbers, mode, gdz, llm, config)

    assessment_prep = _text_has_keyword(
        task.homework_text, config.get("assessment_keywords") or []
    )

    category = classify_task(task, config=config)

    if category == "write_task":
        # Готовий текст твору/есе — окремий mode="write" незалежно від
        # mode, з яким викликали solve_task (Telegram завжди просить
        # "explain", але для твору потрібен інший system prompt).
        llm = llm or LlmSolver(config=config)
        answer = llm.solve(task, mode="write", assessment_prep=assessment_prep)
        return {"source": "llm", "answer": answer, "confidence": "low"}

    if category != "textbook":
        llm = llm or LlmSolver(config=config)
        answer = llm.solve(task, mode=mode, assessment_prep=assessment_prep)
        return {"source": "llm", "answer": answer, "confidence": "low"}

    # 1. Primary: реальна умова задачі з PDF підручника, якщо вказаний.
    task_for_llm = task
    used_pdf_condition = False
    condition = _condition_from_textbook_pdf(task, config)
    if condition:
        task_for_llm = replace(task, exercise_source_text=condition)
        used_pdf_condition = True
        logger.info(
            "Умову задачі витягнуто з PDF підручника — LLM отримає повний текст."
        )
    else:
        logger.info(
            "Умова з PDF не знайдена/підручник не вказаний для '%s' — LLM "
            "працює з коротким записом із щоденника.",
            task.subject,
        )

    # 2. LLM-розв'язок і пошук відповіді в ГДЗ (Book4Source, config.yaml ->
    #    gdz_sources) ЗАВЖДИ йдуть ОДНОЧАСНО (concurrent.futures), не
    #    послідовно "спершу LLM, потім ГДЗ якщо щось не так" — саме так і
    #    було задумано (звірка LLM<->ГДЗ), просто раніше gdz_sources не
    #    існувало як окреме джерело (Book4Source дивився в textbooks, які
    #    тепер вказують на PDF, а не на сторінки ГДЗ) — тому GDZ.search()
    #    фактично завжди повертав None, і звірка ніколи не траплялась.
    gdz = gdz or Book4Source(config=config)
    llm = llm or LlmSolver(config=config)

    def _timed_gdz_search():
        with _timed(
            "gdz_search", subject=task.subject, exercise=(task.book_page or {}).get("exercise")
        ):
            return gdz.search(task.subject, task.book_page)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        gdz_future = executor.submit(_timed_gdz_search)
        llm_future = executor.submit(llm.solve, task_for_llm, mode, assessment_prep)
        try:
            found = gdz_future.result()
        except Exception as exc:  # ГДЗ ніколи не має валити весь solve_task
            logger.info("Book4Source.search() впав: %s", exc)
            found = None
        llm_answer = llm_future.result()  # LlmSolverError тут пролітає нагору як і раніше

    source = "pdf+llm" if used_pdf_condition else "llm"

    if not found:
        return {"source": source, "answer": llm_answer, "confidence": "low"}

    gdz_image_url = found.get("source_image_url")
    if gdz_image_url:
        result = _compare_with_gdz_scan(task_for_llm, gdz_image_url, llm, llm_answer, source)
        result["source_image_url"] = gdz_image_url
        return result

    try:
        matches = _answers_match(llm, task_for_llm, llm_answer, found["raw_answer"])
    except LlmSolverError as exc:
        logger.warning("Звірка LLM<->ГДЗ (текст) не вдалась: %s — вважаю розбіжністю.", exc)
        matches = False

    if matches:
        return {
            "source": "gdz+llm",
            "answer": f"{llm_answer}\n\n✅ Звірено з ГДЗ — відповіді збігаються.",
            "confidence": "high",
        }
    return {
        "source": "gdz+llm",
        "answer": (
            "⚠️ Розбіжність між LLM і ГДЗ — потрібна перевірка людиною.\n\n"
            f"LLM-пояснення:\n{llm_answer}\n\n"
            f"Відповідь з ГДЗ:\n{found['raw_answer']}"
        ),
        "confidence": "low",
    }


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #

def _tasks_from_diary(days: list[dict]) -> list[Task]:
    tasks = []
    for day in days:
        for lesson in day.get("lessons", []):
            hw = lesson.get("homework")
            if hw:
                tasks.append(Task(subject=lesson.get("subject") or "?", homework_text=hw))
    return tasks


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Розв'язувач ДЗ по щоденнику nz.ua.")
    parser.add_argument("--diary", required=True, help="JSON-файл з виводу nz_client.py.")
    parser.add_argument(
        "--mode", choices=["answer", "explain"], default="explain", help="Режим LLM-розв'язку."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Лише класифікувати завдання (textbook/creative), не викликати LLM.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Детальний лог (DEBUG).")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    diary_path = Path(args.diary)
    if not diary_path.exists():
        print(f"Помилка: файл не знайдено: {diary_path}", file=sys.stderr)
        return 2

    try:
        days = json.loads(diary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Помилка: {diary_path} не валідний JSON: {exc}", file=sys.stderr)
        return 2

    tasks = _tasks_from_diary(days)
    if not tasks:
        print("У щоденнику не знайдено жодного ДЗ.")
        return 0

    config = _load_config()
    llm: Optional[LlmSolver] = None

    try:
        for task in tasks:
            category = classify_task(task, config=config)
            print(f"\n=== {task.subject} [{category}] ===")
            print(f"ДЗ: {task.homework_text}")

            if args.dry_run:
                print(f"book_page: {task.book_page}")
                continue

            if llm is None and category in ("textbook", "creative"):
                try:
                    llm = LlmSolver()
                except LlmSolverError as exc:
                    print(f"Помилка: {exc}", file=sys.stderr)
                    return 1

            try:
                result = solve_task(task, mode=args.mode, llm=llm, config=config)
            except LlmSolverError as exc:
                print(f"Помилка розв'язку: {exc}", file=sys.stderr)
                continue

            print(f"[{result['source']}, confidence={result['confidence']}]")
            print(result["answer"])
    finally:
        # Закриваємо всі відкриті Playwright-профілі (BrowserChatProvider),
        # щоб не лишати процеси Chromium висіти після завершення скрипта.
        if llm is not None:
            for provider in llm._provider_cache.values():
                close = getattr(provider, "close", None)
                if callable(close):
                    close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
