"""
Клієнт для парсингу щоденника nz.ua.

nz.ua — SSR-сайт на Yii2, без окремого JSON API (див. nz-api-notes.md).
Дані щоденника віддаються готовим HTML на GET /schedule/diary, тому цей
модуль логіниться звичайною HTML-формою і далі парсить сторінки через
BeautifulSoup за селекторами з nz-api-notes.md, розділ 4.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
from datetime import date as _date, timedelta as _timedelta
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("nz_client")

BASE_URL = "https://nz.ua"


class NzLoginError(Exception):
    """Логін на nz.ua не вдався (невірні дані, протухла сесія, зміна форми)."""


class NzParseError(Exception):
    """Не вдалось завантажити або розпарсити сторінку nz.ua."""


class NzClient:
    def __init__(self, base_url: str = BASE_URL, timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                )
            }
        )

    # ------------------------------------------------------------------ #
    # Автентифікація
    # ------------------------------------------------------------------ #
    def login(self, username: str, password: str) -> None:
        """
        Логінить сесію на nz.ua.

        Форма логіну не задокументована як окремий ендпоінт (модалка на "/"),
        тому action/поля не хардкодяться — форма з password-інпутом
        знаходиться і парситься динамічно, щоб пережити дрібні зміни розмітки.
        """
        try:
            resp = self.session.get(self.base_url + "/", timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise NzLoginError(f"Не вдалось завантажити nz.ua: {exc}") from exc

        soup = BeautifulSoup(resp.text, "html.parser")
        form = self._find_login_form(soup)
        if form is None:
            raise NzLoginError(
                "Не знайшов форму логіну (input[type=password]) на головній "
                "сторінці nz.ua — можливо сайт змінив розмітку."
            )

        action = form.get("action") or "/"
        method = (form.get("method") or "post").lower()
        if action.startswith("http"):
            post_url = action
        else:
            post_url = self.base_url + (action if action.startswith("/") else "/" + action)

        payload: dict[str, str] = {}
        user_field = pass_field = None
        for inp in form.find_all("input"):
            name = inp.get("name")
            if not name:
                continue
            itype = (inp.get("type") or "text").lower()
            if itype == "password":
                pass_field = name
                continue
            if itype in ("text", "email") and user_field is None:
                user_field = name
                continue
            payload[name] = inp.get("value", "")

        if user_field is None or pass_field is None:
            raise NzLoginError(
                "Не зміг визначити поля username/password у формі логіну — "
                "розмітка сторінки могла змінитись."
            )

        payload[user_field] = username
        payload[pass_field] = password

        # Приховані поля форми вже потрапили в payload вище; про всяк
        # випадок підстрахуємось CSRF-мета-тегами, якщо форма їх не несла.
        csrf_meta = soup.find("meta", {"name": "csrf-token"})
        csrf_param = soup.find("meta", {"name": "csrf-param"})
        if csrf_meta and csrf_param:
            payload.setdefault(
                csrf_param.get("content", "_csrf"), csrf_meta.get("content", "")
            )

        try:
            if method == "get":
                resp = self.session.get(post_url, params=payload, timeout=self.timeout)
            else:
                resp = self.session.post(post_url, data=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise NzLoginError(f"Запит логіну не пройшов: {exc}") from exc

        if not self._response_is_authenticated(resp.text):
            raise NzLoginError(
                "Логін не вдався — перевір ім'я користувача/пароль "
                "(сервер повернув сторінку логіну замість кабінету)."
            )

        logger.info("Логін на nz.ua успішний.")

    @staticmethod
    def _find_login_form(soup: BeautifulSoup):
        for form in soup.find_all("form"):
            if form.find("input", {"type": "password"}):
                return form
        return None

    @staticmethod
    def _response_is_authenticated(html: str) -> bool:
        soup = BeautifulSoup(html, "html.parser")
        return soup.find("input", {"type": "password"}) is None

    # ------------------------------------------------------------------ #
    # Cookies
    # ------------------------------------------------------------------ #
    def save_cookies(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self.session.cookies, fh)
        logger.info("Cookies збережено в %s", path)

    def load_cookies(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        try:
            with path.open("rb") as fh:
                cookies = pickle.load(fh)
        except (pickle.UnpicklingError, EOFError, AttributeError) as exc:
            logger.warning("Не вдалось прочитати файл cookies %s: %s", path, exc)
            return False
        self.session.cookies.update(cookies)
        return True

    def is_logged_in(self) -> bool:
        """Легка перевірка валідності сесії (один GET щоденника)."""
        try:
            resp = self.session.get(self.base_url + "/schedule/diary", timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException:
            return False
        return self._response_is_authenticated(resp.text)

    # ------------------------------------------------------------------ #
    # Щоденник
    # ------------------------------------------------------------------ #
    def get_diary_week(self, start_date: str, student_id: str) -> list[dict]:
        """
        GET /schedule/diary?start_date=...&student_id=...&type=for-school

        Повертає список днів: [{date, day_label, lessons: [...]}, ...], у
        порядку понеділок -> неділя (7 елементів). start_date МАЄ бути
        понеділком того тижня (усі виклики в проєкті дотримуються цього) —
        реальна дата кожного дня рахується як start_date + індекс у списку,
        бо сторінка nz.ua не віддає дату окремо для кожного .diary-item
        (лише текстову day_label на кшталт "вівторок, 1 вересня — сьогодні").
        Структура lessons — nz-api-notes.md, розділ 4.
        """
        url = self.base_url + "/schedule/diary"
        params = {
            "start_date": start_date,
            "student_id": student_id,
            "type": "for-school",
        }
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise NzParseError(f"Не вдалось завантажити щоденник: {exc}") from exc

        if not self._response_is_authenticated(resp.text):
            raise NzLoginError(
                "Сесія неактивна — сервер повернув сторінку логіну замість "
                "щоденника. Залогінься знову (--login)."
            )

        soup = BeautifulSoup(resp.text, "html.parser")
        diary_items = soup.select(".diary-item")
        if not diary_items:
            logger.warning(
                "Не знайдено жодного .diary-item на сторінці щоденника — "
                "або на цей тиждень немає даних, або сайт змінив розмітку."
            )
            return []

        monday = _date.fromisoformat(start_date)
        return [
            self._parse_diary_item(item, (monday + _timedelta(days=idx)).isoformat())
            for idx, item in enumerate(diary_items)
        ]

    def _parse_diary_item(self, item, item_date: str) -> dict:
        title_el = item.select_one(".diary-item__title")
        if title_el is None:
            logger.warning(".diary-item__title не знайдено — day_label буде None.")
        day_label = title_el.get_text(strip=True) if title_el else None

        lessons = [self._parse_diary_box(box) for box in item.select(".diary-box")]

        return {"date": item_date, "day_label": day_label, "lessons": lessons}

    def _parse_diary_box(self, box) -> dict:
        lesson: dict = {
            "number": None,
            "time": None,
            "subject": None,
            "cabinet": None,
            "topic": None,
            "homework": None,
            "grade": None,
            "substitution": False,
            "homework_pending_link": None,
        }

        num_el = box.select_one(".diary-item__num")
        if num_el:
            lesson["number"] = num_el.get_text(strip=True).rstrip(".")
        else:
            logger.warning(".diary-item__num не знайдено в одному з уроків.")

        time_el = box.select_one(".diary-item__time")
        if time_el:
            raw = time_el.get_text(separator="|", strip=True)
            parts = [p for p in raw.split("|") if p]
            lesson["time"] = "-".join(parts) if parts else raw
        else:
            logger.warning(".diary-item__time не знайдено в одному з уроків.")

        label_el = box.select_one(".diary-item__label")
        if label_el:
            substitution_span = label_el.find("span")
            if substitution_span and "заміна" in substitution_span.get_text(strip=True).lower():
                lesson["substitution"] = True
                subject_text = label_el.get_text(strip=True).replace(
                    substitution_span.get_text(strip=True), ""
                ).strip()
                lesson["subject"] = subject_text or None
            else:
                lesson["subject"] = label_el.get_text(strip=True) or None
        else:
            logger.warning(".diary-item__label (предмет) не знайдено в одному з уроків.")

        cabinet_el = box.select_one(".diary-item__cabinet")
        if cabinet_el:
            lesson["cabinet"] = cabinet_el.get_text(strip=True)

        for row in box.select(".diary-lesson-row"):
            text_el = row.select_one(".diary-lesson-text")
            row_text = text_el.get_text(strip=True) if text_el else ""

            lowered = row_text.lower()
            if lowered.startswith("поточна:"):
                lesson["topic"] = row_text.split(":", 1)[1].strip()
            elif lowered.startswith("д/з:") or lowered.startswith("дз:"):
                lesson["homework"] = row_text.split(":", 1)[1].strip()
            elif row_text and lesson["topic"] is None:
                lesson["topic"] = row_text

            grade_el = row.select_one(".diary-add-green")
            if grade_el:
                match = re.search(r"point-(\d+)", " ".join(grade_el.get("class", [])))
                if match:
                    lesson["grade"] = int(match.group(1))
                else:
                    txt = grade_el.get_text(strip=True)
                    if txt.isdigit():
                        lesson["grade"] = int(txt)
                    else:
                        logger.warning(
                            "Знайдено .diary-add-green без розпізнаваної оцінки "
                            "(ні point-N в класі, ні числа в тексті '%s').",
                            txt,
                        )

            if row.select_one(".diary-add-red"):
                link_el = row.select_one("a[href]")
                if link_el:
                    lesson["homework_pending_link"] = link_el["href"]
                else:
                    logger.warning(
                        "Знайдено .diary-add-red (незакрите ДЗ) без <a href> поруч."
                    )

        return lesson


# ---------------------------------------------------------------------- #
# Евристика "підручник / сторінка / вправа" з тексту ДЗ
# ---------------------------------------------------------------------- #

# Друга група — опційний кінець діапазону ("ст.3-10" -> сторінки 3, кінець
# 10). Реальний кейс ДЗ без жодного §N/номера вправи ("Опрацювати
# матеріал на ст.3-10, напис. план") — extract_book_page() раніше віддавав
# лише "3", solve_task взагалі не мав шляху витягти текст (тільки
# exercise/paragraph гілки) і завжди йшов на LLM без умови — баг
# знайдений і виправлений 2026-09-09 (extract_page_range_text нижче).
_PAGE_PATTERNS = [
    re.compile(r"\bстор(?:інка|\.)?\s*(\d+)(?:\s*[-–—]\s*(\d+))?", re.IGNORECASE),
    re.compile(r"\bст\.?\s*(\d+)(?:\s*[-–—]\s*(\d+))?", re.IGNORECASE),
    re.compile(r"\bс\.\s*(\d+)(?:\s*[-–—]\s*(\d+))?", re.IGNORECASE),
]

_EXERCISE_PATTERNS = [
    re.compile(r"\bвправ[ауи]?\.?\s*№?\s*(\d+(?:\.\d+)?[а-яіїєa-z]?)", re.IGNORECASE),
    re.compile(r"\bвпр\.?\s*(\d+(?:\.\d+)?[а-яіїєa-z]?)", re.IGNORECASE),
    re.compile(r"№\s*(\d+(?:\.\d+)?[а-яіїєa-z]?)", re.IGNORECASE),
]

# Продовження списку номерів ПІСЛЯ вже знайденого exercise (extract_all_
# exercises нижче): "№27.10. 27.12, 27.13, 27.16" — реальний кейс з
# щоденника (перевірено наживо 2026-09-07/08), де кожен наступний номер
# відділений комою АБО крапкою-пробілом, і "№" повторюється не завжди
# (лише перший номер має "№", решта — голі числа). Тому "№" тут опційний.
# УВАГА: без "^" навмисно — .match(text, pos) вже анкерує пошук рівно на
# pos; "^" тут би вимагав pos == 0 (початок УСЬОГО рядка, не позиції) і
# ніколи не спрацьовував би для другого/третього номера в списку (реальний
# баг, знайдений і виправлений тут-таки).
_EXERCISE_LIST_TAIL_RE = re.compile(
    r"\s*[,.]\s*№?\s*(\d+(?:\.\d+)?[а-яіїєa-z]?)", re.IGNORECASE
)

# "§ 5", "параграф 12", "пункт 27" — посилання на ЦІЛИЙ параграф/пункт
# (типово для "Опрацювати §N"/"Прочитати параграф N", без конкретного
# номера вправи). ВАЖЛИВО: раніше цей випадок не розпізнавався взагалі —
# extract_book_page() повертав None, book_page був None, і LLM ніколи
# навіть не намагався звернутись до textbooks[subject], хоча URL міг бути
# налаштований — саме тому бот писав "не маю тексту параграфа" навіть коли
# PDF підручника був підключений (баг знайдений і виправлений 2026-09-06).
_PARAGRAPH_PATTERNS = [
    re.compile(r"§\s*(\d+)"),
    re.compile(r"\bпараграф\w*\.?\s*№?\s*(\d+)", re.IGNORECASE),
    re.compile(r"\bпункт\w*\.?\s*№?\s*(\d+)", re.IGNORECASE),
]


def extract_book_page(homework_text: str) -> Optional[dict]:
    """
    Евristика regex для сторінки/вправи/параграфа в тексті ДЗ.

    Повертає {"page", "exercise", "paragraph", "source_text"} або None,
    якщо жоден патерн не спрацював — тоді викликач може впасти на
    LLM-фолбек, маючи оригінальний homework_text (в полі source_text)
    незмінним. "exercise" — конкретна вправа (шукається як пункт умови в
    PDF), "paragraph" — посилання на цілий §N без номера вправи (шукається
    як весь текст параграфа для читання/вивчення) — це РІЗНІ типи
    вилучення в textbook_source.py, тому лежать в окремих полях.
    """
    if not homework_text:
        return None

    page_match = next(
        (m for pat in _PAGE_PATTERNS if (m := pat.search(homework_text))), None
    )
    page = page_match.group(1) if page_match else None
    page_end = page_match.group(2) if page_match else None
    exercise = next(
        (m.group(1) for pat in _EXERCISE_PATTERNS if (m := pat.search(homework_text))),
        None,
    )
    paragraph = next(
        (m.group(1) for pat in _PARAGRAPH_PATTERNS if (m := pat.search(homework_text))),
        None,
    )

    if page is None and exercise is None and paragraph is None:
        return None

    return {
        "page": page,
        "page_end": page_end,
        "exercise": exercise,
        "paragraph": paragraph,
        "source_text": homework_text,
    }


def extract_all_exercises(homework_text: str) -> list[str]:
    """
    Усі номери вправ, згадані в ДЗ через кому/крапку (напр. "розв'язати
    №27.10. 27.12, 27.13, 27.16" -> ["27.10", "27.12", "27.13", "27.16"]),
    на відміну від extract_book_page(), яка бере лише ПЕРШИЙ номер.

    Використовується solve_task() (solver.py), щоб розв'язати й звірити
    кожен номер ОКРЕМО замість змішування кількох різних вправ в одну
    умову/відповідь (баг знайдений на реальному ДЗ з геометрії
    2026-09-08). Один номер (чи жодного) -> список з 0-1 елементів, solve_task
    тоді працює як раніше, без розбиття.
    """
    if not homework_text:
        return []

    first = next(
        (m for pat in _EXERCISE_PATTERNS if (m := pat.search(homework_text))), None
    )
    if first is None:
        return []

    numbers = [first.group(1)]
    pos = first.end()
    while True:
        m = _EXERCISE_LIST_TAIL_RE.match(homework_text, pos)
        if not m:
            break
        numbers.append(m.group(1))
        pos = m.end()

    # dict.fromkeys — унікальні номери зі збереженням порядку появи.
    return list(dict.fromkeys(numbers))


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Парсер щоденника nz.ua (SSR, без JSON API).")
    parser.add_argument(
        "--login", action="store_true", help="Примусово залогінитись (ігнорувати збережені cookies)."
    )
    parser.add_argument("--week", required=True, help="Понеділок тижня, формат YYYY-MM-DD.")
    parser.add_argument(
        "--student-id", default=None, help="ID учня (інакше NZ_STUDENT_ID з .env)."
    )
    parser.add_argument("--username", default=None, help="Логін nz.ua (інакше NZ_USERNAME з .env).")
    parser.add_argument("--password", default=None, help="Пароль nz.ua (інакше NZ_PASSWORD з .env).")
    parser.add_argument(
        "--cookies", default=".nz_cookies.pkl", help="Файл для збереження/читання cookies сесії."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Детальний лог (DEBUG).")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    username = args.username or os.environ.get("NZ_USERNAME")
    password = args.password or os.environ.get("NZ_PASSWORD")
    student_id = args.student_id or os.environ.get("NZ_STUDENT_ID")

    if not student_id:
        print("Помилка: не задано --student-id ані NZ_STUDENT_ID у .env", file=sys.stderr)
        return 2

    client = NzClient()

    have_session = False
    if not args.login:
        have_session = client.load_cookies(args.cookies) and client.is_logged_in()
        if not have_session:
            logger.info("Немає валідної збереженої сесії — потрібен логін.")

    if not have_session:
        if not username or not password:
            print(
                "Помилка: потрібен логін, але не задано --username/--password "
                "ані NZ_USERNAME/NZ_PASSWORD у .env",
                file=sys.stderr,
            )
            return 2
        try:
            client.login(username, password)
        except NzLoginError as exc:
            print(f"Помилка логіну: {exc}", file=sys.stderr)
            return 1
        client.save_cookies(args.cookies)

    try:
        days = client.get_diary_week(args.week, student_id)
    except (NzParseError, NzLoginError) as exc:
        print(f"Помилка: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(days, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
