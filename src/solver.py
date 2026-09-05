"""
Розв'язувач ДЗ по щоденнику nz.ua.

Бере уроки з JSON-виводу nz_client.py (get_diary_week), класифікує кожне ДЗ
як "textbook" (є конкретна сторінка/вправа з підручника — спершу шукаємо в
ГДЗ) чи "creative" (твір/есе — одразу LLM), і повертає розв'язок.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Optional

import yaml

from nz_client import extract_book_page
from textbook_source import DEFAULT_CACHE_DIR, download_textbook, extract_exercise_condition

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("solver")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ -> корінь проєкту
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DB_PATH = PROJECT_ROOT / "nz_solver.db"


class SolverError(Exception):
    """Базова помилка цього модуля."""


class LlmSolverError(SolverError):
    """Виклик LLM не вдався (мережа, авторизація, ліміти)."""


class BrowserSessionExpiredError(LlmSolverError):
    """Сесія браузерного профілю злетіла — бачу форму логіну замість чату."""


class RateLimitError(LlmSolverError):
    """
    Провайдер повернув rate-limit/quota помилку (429/RESOURCE_EXHAUSTED).
    Окремий тип від LlmSolverError навмисно — LlmSolver.solve() ловить
    САМЕ цей тип, щоб автоматично пробувати наступного провайдера з
    config.yaml -> llm.fallback_order. Інші помилки (авторизація, мережа,
    невідомий провайдер) не мають такого автоматичного failover'у — вони
    зазвичай означають проблему конфігурації, яку тихий fallback лише
    замаскував би.
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
        textbooks = self.config.get("textbooks") or {}
        book_url = textbooks.get(subject)
        if not book_url:
            logger.warning(
                "Автор підручника не вказано для '%s' (config.yaml -> "
                "textbooks) — іде напряму на LLM.",
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
            if self._text_matches(text, book_page):
                return href
        return None

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
    def _ordinal_key(ordinal: tuple) -> tuple:
        major, minor = ordinal
        return (major, minor if minor is not None else -1)

    def _text_matches(self, text: str, book_page: dict) -> bool:
        page = book_page.get("page")
        exercise = book_page.get("exercise")

        m = self._PAGE_EXERCISE_RE.search(text)
        if m and page:
            try:
                if int(m.group(1)) == int(page):
                    return exercise is None or str(m.group(2)) == str(exercise).strip()
            except ValueError:
                pass

        if exercise:
            target = self._parse_ordinal(str(exercise))
            if target is not None:
                tk = self._ordinal_key(target)
                rm = self._RANGE_RE.search(text)
                if rm:
                    lo, hi = self._parse_ordinal(rm.group(1)), self._parse_ordinal(rm.group(2))
                    if lo is not None and hi is not None:
                        if self._ordinal_key(lo) <= tk <= self._ordinal_key(hi):
                            return True
                else:
                    sm = self._SINGLE_NUM_RE.search(text)
                    if sm:
                        single = self._parse_ordinal(sm.group(1))
                        if single is not None and single == target:
                            return True
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

_EXPLAIN_SYSTEM = """\
Ти — репетитор, який допомагає учню 10 класу з домашнім завданням через Telegram.

Обсяг відповіді:
- За замовчуванням пиши КОРОТКО: 3–6 речень або пунктів на завдання. Ціль —
  дати учню зрозумілий орієнтир що робити, а не переписати підручник.
- Розгорнуте покрокове пояснення (більше кроків, приклади) — лише якщо
  завдання СПРАВДІ складне (розв'язання рівняння/нерівності/системи,
  задача з кількома невідомими, доведення).
- Для організаційних завдань ("повторити", "прочитати", "вивчити терміни",
  "опрацювати параграф") дивись у "Завдання"/"Довідково" в user-повідомленні:
  - якщо там є РЕАЛЬНИЙ зміст параграфа/термінів (текст, а не короткий
    запис із щоденника) — впиши конкретні терміни/факти з нього.
  - якщо реального змісту НЕМАЄ (відомий лише короткий запис на кшталт
    "Опрацювати §5, вивчити терміни") — ЧЕСНО скажи одним реченням "Не маю
    тексту параграфа, щоб виписати конкретні терміни" і дай лише загальний
    метод роботи (2–3 речення: як ефективно опрацювати параграф/вивчити
    терміни). НІКОЛИ не вигадуй конкретні терміни/факти, яких не знаєш.
- Все одно поясни логіку, а не дай голу відповідь без жодного слова —
  просто стисло, без зайвого.

Стиль:
- НІКОЛИ не розігруй діалог на кшталт "Я (репетитор):" / "Учень:" —
  звертайся до учня прямо, без сценки.
- НІКОЛИ не використовуй Markdown-таблиці (рядки з "|"). Замість таблиці —
  звичайний список ("•" або "1.", "2.").

Розмітка виводу — ПРОСТИЙ MARKDOWN (не HTML, теги на кшталт <b> тут писати
НЕ треба — конвертацію в Telegram-розмітку робить окремий код, не ти):
- **жирний текст** для ключових термінів/висновків.
- _курсив_ для другорядного наголосу (рідко, за потреби).
- `код`/формули у зворотних лапках, якщо доречно.
- Списки — звичайним текстом: кожен пункт на новому рядку, що починається
  з "•" або "1.", "2." — БЕЗ Markdown-таблиць (рядків з "|") і без ### заголовків.

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
        self.client = openai.OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
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
        self.client = genai.Client(api_key=key)
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
            raise LlmSolverError(f"Gemini API: серверна помилка: {exc}") from exc
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

        Якщо ВСІ провайдери в ланцюжку впали з rate-limit — кидає
        LlmSolverError з коротким людським повідомленням (повні деталі
        останньої помилки йдуть лише в лог, не в текст винятку, щоб їх не
        побачив користувач у Telegram).
        """
        if mode == "write":
            system = _WRITE_SYSTEM
        else:
            system = _EXPLAIN_SYSTEM if mode == "explain" else _ANSWER_SYSTEM
            if assessment_prep and mode == "explain":
                system += _ASSESSMENT_PREP_ADDENDUM
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
                "" if idx == 0 else " [fallback після rate-limit]",
            )
            try:
                return provider.complete(system, user_content)
            except RateLimitError as exc:
                logger.warning(
                    "Провайдер '%s' впав з rate-limit/quota: %s — пробую наступного в "
                    "ланцюжку %s.",
                    name,
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
    cached = _ocr_cache_get(image_url, kind=kind)
    if cached is not None:
        logger.info("OCR (%s): знайдено в кеші для %s.", kind, image_url)
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

    client = genai.Client(api_key=key)
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
    verdict = provider.complete(_COMPARE_SYSTEM, prompt)
    return verdict.strip().lower().startswith("так")


# ---------------------------------------------------------------------- #
# solve_task
# ---------------------------------------------------------------------- #

def _resolve_scan_result(
    task: Task,
    found: dict,
    mode: Literal["answer", "explain"],
    llm: LlmSolver,
    assessment_prep: bool = False,
) -> dict:
    image_url = found["source_image_url"]

    # Спершу пробуємо розпізнати саму УМОВУ задачі з того ж скану, щоб LLM
    # розв'язував реальну задачу, а не вгадував з короткого запису типу
    # "Виконати №1.13".
    condition_text = None
    try:
        condition = _ocr_scan(
            image_url, prompt=_OCR_CONDITION_PROMPT, kind="condition"
        ).strip()
        if condition and condition != _NO_CONDITION_MARKER:
            condition_text = condition
    except LlmSolverError as exc:
        logger.info(
            "OCR умови не вдався для %s: %s — LLM працюватиме з коротким "
            "записом із щоденника.",
            image_url,
            exc,
        )

    task_for_llm = task
    if condition_text:
        task_for_llm = replace(task, exercise_source_text=condition_text)
        logger.info("OCR розпізнав умову задачі — LLM отримає повний текст умови.")
    else:
        logger.info(
            "Умова на скані не знайдена/не розпізнана — LLM працює з "
            "коротким записом із щоденника."
        )

    llm_answer = llm.solve(task_for_llm, mode=mode, assessment_prep=assessment_prep)

    try:
        gdz_text = _ocr_scan(image_url)
    except LlmSolverError as exc:
        logger.warning(
            "OCR відповіді не вдався для %s: %s — повертаю LLM-відповідь без звірки.",
            image_url,
            exc,
        )
        return {"source": "llm", "answer": llm_answer, "confidence": "low"}

    try:
        matches = _answers_match(llm, task_for_llm, llm_answer, gdz_text)
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
    Пробує витягти реальну умову задачі з PDF підручника (config.yaml ->
    textbooks[subject] — тепер це URL PDF, не ГДЗ-сторінки). None, якщо
    підручник не вказано, немає номера вправи, чи текст не знайшовся —
    ніколи не кидає виняток (мережа/PDF — best-effort primary джерело,
    solve_task завжди має чим фолбекнутись).
    """
    if not task.book_page or not task.book_page.get("exercise"):
        return None

    textbook_url = (config.get("textbooks") or {}).get(task.subject)
    if not textbook_url:
        return None

    cache_dir = config.get("textbook_cache_dir") or str(DEFAULT_CACHE_DIR)
    try:
        pdf_path = download_textbook(textbook_url, cache_dir)
    except Exception as exc:
        logger.info("Не вдалось завантажити підручник %s: %s", textbook_url, exc)
        return None

    try:
        return extract_exercise_condition(
            pdf_path, task.book_page["exercise"], task.book_page.get("page")
        )
    except Exception as exc:
        logger.info("Не вдалось витягти умову з %s: %s", pdf_path, exc)
        return None


def solve_task(
    task: Task,
    mode: Literal["answer", "explain"] = "explain",
    gdz: Optional[GdzSource] = None,
    llm: Optional[LlmSolver] = None,
    config: Optional[dict] = None,
) -> dict:
    """
    Повертає {"source": "pdf+llm"|"gdz"|"gdz+llm"|"llm"|"skipped", "answer": str,
    "confidence": "high"|"low"}.
    """
    config = config if config is not None else _load_config()

    if is_review_only(task.homework_text, config=config):
        return {
            "source": "skipped",
            "answer": "Просто повторити конспект/підручник — детальний розбір не потрібен.",
            "confidence": "high",
        }

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

    # 2. Другорядно (опційно): звірка фінальної відповіді зі сканом ГДЗ.
    #    ПРИМІТКА: якщо textbooks[subject] тепер містить URL PDF (а не
    #    сторінки ГДЗ), Book4Source шукатиме посилання всередині PDF-байтів
    #    і коректно поверне None — ця гілка фактично неактивна, поки для
    #    Book4Source не заведуть окреме джерело URL.
    gdz = gdz or Book4Source(config=config)
    found = gdz.search(task.subject, task.book_page)
    if found:
        if found.get("source_image_url"):
            llm = llm or LlmSolver(config=config)
            return _resolve_scan_result(
                task_for_llm, found, mode, llm, assessment_prep=assessment_prep
            )
        return {"source": "gdz", "answer": found["raw_answer"], "confidence": "high"}

    llm = llm or LlmSolver(config=config)
    answer = llm.solve(task_for_llm, mode=mode, assessment_prep=assessment_prep)
    source = "pdf+llm" if used_pdf_condition else "llm"
    return {"source": source, "answer": answer, "confidence": "low"}


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
