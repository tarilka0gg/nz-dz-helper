"""
Витягування умов задач напряму з PDF підручника (pidruchnyk.com.ua тощо) —
альтернатива/доповнення до OCR зі сканів ГДЗ (solver.py). PDF зазвичай дає
чистіший текст (без плутанини на кшталт "2n" замість "2ⁿ", яку ловить OCR
зображень), тому це primary-джерело умови задачі для LlmSolver.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("textbook_source")
timing_logger = logging.getLogger("solver.timing")

# Спільна БД з solver.py (nz_solver.db, корінь проєкту) — нова таблиця
# textbook_index туди ж, щоб не плодити окремих файлів. Шлях зібраний
# незалежно (не імпортом з solver.py) — інакше вийшло б циклічне
# імпортування (solver.py вже імпортує цей модуль).
INDEX_DB_PATH = Path(__file__).resolve().parent.parent / "nz_solver.db"

# Кілька предметів (напр. Алгебра й Геометрія) можуть посилатись на ОДИН і
# той самий URL/файл — при паралельному розв'язуванні (telegram_bot.py)
# перший запит для обох міг би одночасно писати в той самий .part-файл і
# зіпсувати завантаження. Лок серіалізує лише сам факт завантаження
# (рідкісний, одноразовий шлях); уже закешовані файли перевіряються без
# нього — швидкий шлях лишається без блокування.
_download_lock = threading.Lock()

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".textbook_cache"  # src/ -> корінь проєкту

# Два формати заголовків вправ, залежно від видавництва:
# 1. Мерзляк (Алгебра/Геометрія): "1.13." — параграф.номер, опційно з
#    позначкою рівня складності (° / • / ••).
# 2. Решта перевірених підручників (Хімія Савчин, Біологія Соболь, Історія
#    Сорочинська, Всесвітня історія Полянський, Укр. мова Авраменко,
#    Географія Гільберг тощо — перевірено наживо 2026-09-06, жоден не
#    використовує крапковий формат): наскрізна нумерація "N." без крапки
#    в номері, теж на початку рядка.
# Обирається за форматом exercise_number: є крапка в номері -> Мерзляк,
# немає -> звичайний формат.
_EXERCISE_HEADER_DOTTED_RE = re.compile(r"(?m)^\s*(\d+\.\d+)\.\s*[°•]{0,2}\s*")
_EXERCISE_HEADER_PLAIN_RE = re.compile(r"(?m)^\s*(\d{1,4})\.\s*[°•]{0,2}\s*")

# Заголовок ЦІЛОГО параграфа "§ N" — для "Опрацювати §N"/"Прочитати
# параграф N" без конкретної вправи (extract_paragraph_text нижче).
_SECTION_HEADER_RE = re.compile(r"(?m)^\s*§\s*(\d+)\.?\s")


def _header_pattern_for(target: str) -> re.Pattern:
    return _EXERCISE_HEADER_DOTTED_RE if "." in target else _EXERCISE_HEADER_PLAIN_RE


# "7а", "12б" тощо — номер вправи з літерою насправді означає "пункт А/Б/..
# всередині вправи 7/12" (перевірено наживо на реальному кейсі: Укр. мова
# §1, "впр. 7а" — у PDF це вправа "7." з підпунктами "А. ..."/"Б. ..."
# всередині неї, а не окрема вправа "7а."). Без цього розбиття
# extract_exercise_condition("7а") ніколи не знаходив заголовка (індекс і
# пряме сканування шукають ЧИСТО цифровий номер), condition завжди був
# None, і виклик (_condition_from_textbook_pdf) фолбечився на
# extract_paragraph_text — цілий параграф з кількома різними вправами
# замість умови саме вправи 7а (баг знайдений і виправлений 2026-09-08).
_LETTERED_SUFFIX_RE = re.compile(r"^(\d+(?:\.\d+)?)([а-яіїєА-ЯІЇЄa-zA-Z])$")

# Підпункти всередині вправи позначені ВЕЛИКОЮ літерою з крапкою на
# початку рядка ("А. Знайдіть...", "Б. створіть..."). Номер вправи в тексті
# ДЗ зазвичай малими літерами ("7а") — порівнюємо в _extract_lettered_subpart
# через .upper().
_SUBITEM_HEADER_RE = re.compile(r"(?m)^\s*([А-ЯІЇЄA-Z])[.\)]\s")


def _split_lettered_exercise(target: str) -> tuple[str, Optional[str]]:
    m = _LETTERED_SUFFIX_RE.match(target)
    if m:
        return m.group(1), m.group(2)
    return target, None


def _extract_lettered_subpart(condition: str, letter: str) -> Optional[str]:
    """
    Звужує повний текст вправи condition до підпункту letter (напр. "а" ->
    шматок від "А. " до наступного "Б. "/кінця). None, якщо підпунктів з
    такою літерою в тексті нема (тоді викликач має показати condition
    цілком — краще ціла вправа, ніж нічого).
    """
    target = letter.upper()
    headers = list(_SUBITEM_HEADER_RE.finditer(condition))
    for i, m in enumerate(headers):
        if m.group(1) == target:
            start = m.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(condition)
            sub = condition[start:end].strip()
            return sub or None
    return None


# Бампаємо цей рядок, якщо міняється будь-який з regex вище (чи логіка
# фільтрації нижче) — старий індекс стає "неактуальним" і перебудовується
# автоматично при наступному зверненні.
_INDEX_REGEX_VERSION = "v2-2026-09-07-toc-filter"

# Рядки змісту ("Вступ.......................4") теж збігаються з
# заголовком "§ N" на початку рядка — без фільтра їхній "текст" (усе до
# наступного такого ж заголовка в самому змісті) потрапляв в індекс і
# затирав СПРАВЖНІЙ параграф. Ознака рядка змісту — крапки-заповнювачі
# одразу після заголовка, до номера сторінки.
_TOC_ENTRY_RE = re.compile(r"^[^\n]{0,120}\.{5,}")


def _looks_like_toc_entry(content: str) -> bool:
    return bool(_TOC_ENTRY_RE.match(content))


def _index_db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(INDEX_DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS textbook_index ("
        "textbook_file TEXT NOT NULL, kind TEXT NOT NULL, number TEXT NOT NULL, "
        "page INTEGER, condition_text TEXT NOT NULL, "
        "PRIMARY KEY (textbook_file, kind, number))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS textbook_index_meta ("
        "textbook_file TEXT PRIMARY KEY, regex_version TEXT NOT NULL, indexed_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS textbook_figure_cache ("
        "textbook_file TEXT NOT NULL, page INTEGER NOT NULL, description TEXT NOT NULL, "
        "PRIMARY KEY (textbook_file, page))"
    )
    return conn


def _index_is_current(conn: sqlite3.Connection, textbook_file: str) -> bool:
    row = conn.execute(
        "SELECT regex_version FROM textbook_index_meta WHERE textbook_file = ?",
        (textbook_file,),
    ).fetchone()
    return row is not None and row[0] == _INDEX_REGEX_VERSION


def _page_locator(pages_text: list[str]):
    """Повертає функцію char_offset_у_full_text -> номер_сторінки (0-based),
    без повторного проходу по тексту на кожен заголовок — бінарний пошук
    по префіксних довжинах сторінок."""
    offsets = []
    pos = 0
    for pt in pages_text:
        offsets.append(pos)
        pos += len(pt) + 1  # +1 за "\n", яким склеюємо сторінки нижче

    def locate(char_index: int) -> int:
        lo, hi = 0, len(offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offsets[mid] <= char_index:
                lo = mid
            else:
                hi = mid - 1
        return lo

    return locate


def _build_index(pdf_path: Path) -> None:
    """
    Один прохід по всьому PDF: знаходить УСІ заголовки вправ (крапкові й
    наскрізні) і параграфів (§N), для кожного бере текст до наступного
    такого ж заголовка й записує все разом у SQLite. Виконується ОДИН РАЗ
    на файл (regex_version у textbook_index_meta визначає, чи актуальний
    уже наявний індекс) — наступні extract_exercise_condition()/
    extract_paragraph_text() читають з таблиці замість повторного
    сканування PDF щоразу (раніше — секунди-хвилина на кожен запит для
    великих книг без page_hint).
    """
    import pdfplumber

    textbook_file = pdf_path.name
    logger.info("Будую індекс вправ/параграфів для %s (це станеться лише раз)...", textbook_file)

    with pdfplumber.open(pdf_path) as pdf:
        pages_text = [p.extract_text() or "" for p in pdf.pages]
    full_text = "\n".join(pages_text)
    page_of = _page_locator(pages_text)

    rows = []
    for header_re, kind in (
        (_EXERCISE_HEADER_DOTTED_RE, "exercise"),
        (_EXERCISE_HEADER_PLAIN_RE, "exercise"),
        (_SECTION_HEADER_RE, "paragraph"),
    ):
        headers = list(header_re.finditer(full_text))
        for i, m in enumerate(headers):
            start = m.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(full_text)
            content = full_text[start:end].strip()
            if not content or _looks_like_toc_entry(content):
                continue
            rows.append((textbook_file, kind, m.group(1), page_of(m.start()), content))

    conn = _index_db_connect()
    try:
        conn.execute("DELETE FROM textbook_index WHERE textbook_file = ?", (textbook_file,))
        # INSERT OR IGNORE (не REPLACE) — навмисно лишаємо ПЕРШЕ входження
        # кожного номера. Заголовки §N/вправ часто повторюються в межах
        # книги (розділи можуть перезапускати нумерацію, або те саме "§ 1"
        # трапляється вдруге деінде) — перше входження, як правило, і є
        # справжній параграф, а не випадковий дублікат далі в тексті.
        conn.executemany(
            "INSERT OR IGNORE INTO textbook_index "
            "(textbook_file, kind, number, page, condition_text) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.execute(
            "INSERT OR REPLACE INTO textbook_index_meta "
            "(textbook_file, regex_version, indexed_at) VALUES (?, ?, ?)",
            (textbook_file, _INDEX_REGEX_VERSION, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    logger.info("Проіндексовано %s: %d записів.", textbook_file, len(rows))


def ensure_indexed(pdf_path: Path) -> None:
    """
    Будує індекс для pdf_path, якщо його ще нема або він застарів (змінився
    _INDEX_REGEX_VERSION після виправлення regex). Безпечно викликати
    повторно — швидкий шлях (один SELECT) якщо вже актуальний. Ніколи не
    кидає виняток назовні (індекс — оптимізація, а не обов'язкова умова:
    якщо побудова не вдалась, extract_* просто підуть прямим скануванням).
    """
    try:
        conn = _index_db_connect()
        try:
            current = _index_is_current(conn, pdf_path.name)
        finally:
            conn.close()
        if not current:
            _build_index(pdf_path)
    except Exception as exc:
        logger.info("Не вдалось побудувати/перевірити індекс для %s: %s", pdf_path, exc)


def _index_lookup(pdf_path: Path, kind: str, number: str) -> Optional[str]:
    conn = _index_db_connect()
    try:
        row = conn.execute(
            "SELECT condition_text FROM textbook_index "
            "WHERE textbook_file = ? AND kind = ? AND number = ?",
            (pdf_path.name, kind, number),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def download_textbook(url: str, cache_dir: str) -> Path:
    """
    Завантажує PDF підручника в локальний кеш (за іменем файлу з URL) і
    повертає шлях. Якщо файл уже є в кеші — мережевий запит не робиться.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    filename = url.rstrip("/").rsplit("/", 1)[-1] or "textbook.pdf"
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    dest = cache_path / filename

    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Підручник уже в кеші: %s", dest)
        ensure_indexed(dest)
        return dest

    with _download_lock:
        # Перевіряємо ЩЕ РАЗ під локом — інший потік (напр. паралельне
        # розв'язування Алгебри й Геометрії з тим самим URL) міг устигнути
        # завантажити файл, поки ми чекали.
        if dest.exists() and dest.stat().st_size > 0:
            logger.info("Підручник уже в кеші (завантажив паралельний потік): %s", dest)
            ensure_indexed(dest)
            return dest

        import requests

        logger.info("Завантажую підручник: %s", url)
        resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30, stream=True)
        resp.raise_for_status()

        tmp_dest = dest.with_suffix(dest.suffix + ".part")
        with tmp_dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 64):
                if chunk:
                    fh.write(chunk)
        tmp_dest.replace(dest)
        logger.info("Збережено: %s (%d байт)", dest, dest.stat().st_size)
        # Індексуємо одразу при першому завантаженні (не лежить лінивим
        # чеканням на перший extract_*-запит) — наступні звернення до цього
        # файлу вже підуть швидким шляхом через SQLite.
        ensure_indexed(dest)
        return dest


def _extract_page_text(pdf, index: int) -> str:
    if 0 <= index < len(pdf.pages):
        return pdf.pages[index].extract_text() or ""
    return ""


def _scan_for_exercise(pdf, indices, target: str, header_re: re.Pattern) -> Optional[str]:
    text = "".join(_extract_page_text(pdf, i) + "\n" for i in indices)
    headers = list(header_re.finditer(text))
    for i, m in enumerate(headers):
        if m.group(1) == target:
            start = m.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            condition = text[start:end].strip()
            return condition or None
    return None


def extract_exercise_condition(
    pdf_path: Path, exercise_number: str, page_hint: Optional[str] = None
) -> Optional[str]:
    """
    Шукає в PDF вправу з номером exercise_number і повертає текст від цього
    номера до наступного такого ж заголовка. None, якщо не знайдено
    (сторінка/номер відсутні, PDF пошкоджений тощо — ніколи не кидає
    виняток назовні).

    Формат заголовка обирається за самим exercise_number: "1.13" (є
    крапка) -> параграф.номер як у Мерзляка; "27" (без крапки) -> звичайна
    наскрізна нумерація "27." на початку рядка, як у більшості інших
    перевірених підручників (Хімія, Біологія, Історія тощо — жоден з них
    не використовує крапковий формат Мерзляка).

    УВАГА до плоского формату (без крапки): це поширеніший, але й
    ризикованіший патерн — "N." на початку рядка може випадково збігтися
    з номером у списку/примітці, що не є заголовком вправи. Тому: 1) якщо
    заданий page_hint, шукаємо СПЕРШУ у вузькому вікні навколо нього
    (значно менше шансів на випадковий збіг) і повертаємо результат саме
    звідти, як тільки він знайшовся; 2) повний скан документа — фолбек,
    лише якщо звужений пошук не спрацював.

    Якщо задано page_hint (друкований номер сторінки підручника) — спершу
    звужуємо пошук до вузького вікна навколо неї, і лише якщо там не
    знайшлось — скануємо весь документ.
    """
    target = str(exercise_number).strip()
    if not target:
        return None

    base, letter = _split_lettered_exercise(target)

    # Швидкий шлях: якщо файл уже проіндексований (ensure_indexed —
    # викликається з download_textbook), номер знаходиться миттєво без
    # повторного сканування PDF.
    t0 = time.monotonic()
    ensure_indexed(pdf_path)
    cached = _index_lookup(pdf_path, "exercise", base)
    if cached is not None:
        timing_logger.info(
            "[timing] pdf_index_hit file=%s number=%s: %.3fs",
            pdf_path.name, base, time.monotonic() - t0,
        )
        if letter:
            return _extract_lettered_subpart(cached, letter) or cached
        return cached

    # Індекс НЕ дав відповіді (промт 44, п.1 — перевірка, що індекс справді
    # використовується) — падаємо на повне сканування PDF. Це МАЄ бути
    # рідкістю (лише коли номера справді нема в книзі, чи build_index
    # чомусь не спрацював); якщо цей лог трапляється часто в реальному
    # /week — індекс з якоїсь причини не покриває книгу, і варто дивитись
    # туди, а не сюди.
    logger.warning(
        "PDF-індекс НЕ дав %s (file=%s) — падаю на повне сканування PDF (повільніше).",
        base, pdf_path.name,
    )

    header_re = _header_pattern_for(base)

    try:
        import pdfplumber
    except ImportError:
        logger.warning("Пакет 'pdfplumber' не встановлено (pip install pdfplumber).")
        return None

    t_scan0 = time.monotonic()
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

            if page_hint:
                try:
                    # Друкована сторінка в цьому підручнику відповідає
                    # pdfplumber-індексу приблизно "printed - 1" (перевірено
                    # наживо) — але це не гарантія для будь-якого PDF, тому
                    # беремо запас +-3 сторінки і завжди фолбечимось на
                    # повний скан, якщо у вікні не знайшлось.
                    approx_index = int(page_hint) - 1
                    window = range(max(0, approx_index - 3), min(total_pages, approx_index + 4))
                    result = _scan_for_exercise(pdf, window, base, header_re)
                    if result:
                        if letter:
                            return _extract_lettered_subpart(result, letter) or result
                        return result
                except ValueError:
                    pass

            result = _scan_for_exercise(pdf, range(total_pages), base, header_re)
            if result and letter:
                return _extract_lettered_subpart(result, letter) or result
            return result
    except Exception as exc:  # пошкоджений/нечитний PDF — не валимо виклик
        logger.info("Не вдалось прочитати %s: %s", pdf_path, exc)
        return None
    finally:
        timing_logger.info(
            "[timing] pdf_full_scan file=%s number=%s: %.3fs",
            pdf_path.name, base, time.monotonic() - t_scan0,
        )


_FIGURE_DESCRIPTION_PROMPT = (
    "На цьому зображенні — рисунок зі сторінки підручника геометрії/математики. "
    "Опиши СХЕМАТИЧНО, що на ньому зображено: яка фігура (трикутник, площина, "
    "многогранник тощо), які точки/відрізки/кути/довжини позначені і як вони "
    "співвідносяться. 2-4 речення, без вступних фраз — одразу опис."
)


def _describe_page_figure(pdf_path: Path, page_index: int) -> Optional[str]:
    """
    Якщо на сторінці page_index є вбудовані зображення (page.images) —
    рендерить сторінку в картинку і просить Gemini Vision стисло описати
    рисунок (актуально для геометрії: умова часто посилається на "рисунок
    27.19", якого сам текстовий шар PDF не містить). Кешується в SQLite
    (textbook_figure_cache) — Vision-виклик максимум раз на (файл,
    сторінка), а не на кожен запит. None якщо на сторінці немає зображень,
    GEMINI_API_KEY не заданий, чи Vision-виклик не вдався — ніколи не
    кидає виняток (це опційне збагачення умови, не критичний шлях).
    """
    t0 = time.monotonic()
    conn = _index_db_connect()
    try:
        row = conn.execute(
            "SELECT description FROM textbook_figure_cache WHERE textbook_file = ? AND page = ?",
            (pdf_path.name, page_index),
        ).fetchone()
        if row:
            timing_logger.info(
                "[timing] vision_figure_cache_hit file=%s page=%d: %.3fs",
                pdf_path.name, page_index, time.monotonic() - t0,
            )
            return row[0]
    finally:
        conn.close()

    try:
        import pdfplumber
    except ImportError:
        return None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not (0 <= page_index < len(pdf.pages)):
                return None
            page = pdf.pages[page_index]
            # Рисунки в цих підручниках здебільшого ВЕКТОРНІ (лінії/криві
            # намальовані PDF-примітивами для чіткості), не растрові
            # картинки — page.images тут майже завжди порожній навіть коли
            # рисунок реально є (перевірено наживо: §27.5 Геометрії, 58
            # curves, 0 images). Тому дивимось на растрові ЧИ векторні
            # елементи разом, з невеликим порогом — щоб не спрацьовувати
            # на випадкову декоративну лінію/рамку.
            has_raster = bool(page.images)
            has_vector_figure = (len(page.lines) + len(page.rects) + len(page.curves)) > 5
            if not has_raster and not has_vector_figure:
                return None
            im = page.to_image(resolution=150)
            import io

            buf = io.BytesIO()
            im.original.save(buf, format="PNG")
            img_bytes = buf.getvalue()
    except Exception as exc:
        logger.info("Не вдалось вирізати малюнок зі сторінки %d %s: %s", page_index, pdf_path, exc)
        return None

    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    t_vision0 = time.monotonic()
    try:
        from google import genai
        from google.genai import types as genai_types

        # timeout=45с — без цього SDK-дефолт 600с; той самий клас бага, що
        # й у solver.py-провайдерів (промт 44), тут теж міг би тримати
        # рендер сторінки/Vision-виклик зависаючим замість швидкого None.
        client = genai.Client(
            api_key=api_key, http_options=genai_types.HttpOptions(timeout=45000)
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                genai_types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                _FIGURE_DESCRIPTION_PROMPT,
            ],
        )
        description = getattr(response, "text", None)
    except Exception as exc:
        logger.info("Vision-опис малюнка не вдався для %s стор.%d: %s", pdf_path, page_index, exc)
        return None
    finally:
        timing_logger.info(
            "[timing] vision_figure_call file=%s page=%d: %.3fs",
            pdf_path.name, page_index, time.monotonic() - t_vision0,
        )

    if not description:
        return None

    conn = _index_db_connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO textbook_figure_cache (textbook_file, page, description) "
            "VALUES (?, ?, ?)",
            (pdf_path.name, page_index, description),
        )
        conn.commit()
    finally:
        conn.close()
    return description


def extract_exercise_condition_with_figure(
    pdf_path: Path, exercise_number: str, page_hint: Optional[str] = None
) -> Optional[str]:
    """
    extract_exercise_condition() + якщо відомо, на якій сторінці індексу
    знайшлась вправа, і на ній є вбудовані зображення — дописує короткий
    Vision-опис малюнка в кінець умови. Для книг без картинок (майже всі,
    крім геометрії) — просто те саме, що extract_exercise_condition().
    """
    target = str(exercise_number).strip()
    if not target:
        return None

    base, letter = _split_lettered_exercise(target)

    ensure_indexed(pdf_path)
    conn = _index_db_connect()
    try:
        row = conn.execute(
            "SELECT condition_text, page FROM textbook_index "
            "WHERE textbook_file = ? AND kind = 'exercise' AND number = ?",
            (pdf_path.name, base),
        ).fetchone()
    finally:
        conn.close()

    if row:
        condition, page = row
        if letter:
            condition = _extract_lettered_subpart(condition, letter) or condition
    else:
        # Не в індексі (рідкісний випадок — ensure_indexed не спрацював) —
        # фолбек без інформації про сторінку, отже й без малюнка.
        condition = extract_exercise_condition(pdf_path, exercise_number, page_hint)
        page = None

    if not condition:
        return None

    if page is not None:
        figure = _describe_page_figure(pdf_path, page)
        if figure:
            condition = f"{condition}\n\n[Опис рисунка зі сторінки підручника]: {figure}"

    return condition


def extract_paragraph_text(
    pdf_path: Path, paragraph_number: str, max_chars: int = 6000
) -> Optional[str]:
    """
    Витягує повний текст параграфа §N (від заголовка до наступного §) —
    для ДЗ типу "Опрацювати §N"/"Прочитати параграф N", де немає
    конкретної вправи, а треба весь зміст параграфа для читання/вивчення.
    Обрізає до max_chars (параграфи можуть бути довгими — LLM для короткої
    поради не треба дослівно все, досить основного змісту як орієнтиру).
    None, якщо параграф не знайдено — виклик ніколи не кидає виняток.
    """
    target = str(paragraph_number).strip()
    if not target:
        return None

    t0 = time.monotonic()
    ensure_indexed(pdf_path)
    cached = _index_lookup(pdf_path, "paragraph", target)
    if cached is not None:
        timing_logger.info(
            "[timing] pdf_index_hit file=%s number=§%s: %.3fs",
            pdf_path.name, target, time.monotonic() - t0,
        )
        return cached[:max_chars]

    # Індекс мав би вже покрити весь документ — сюди доходимо лише якщо
    # побудова індексу з якоїсь причини не вдалась (ensure_indexed ловить
    # свої винятки й просто нічого не будує). Прямий скан як останній шанс.
    logger.warning(
        "PDF-індекс НЕ дав §%s (file=%s) — падаю на повне сканування PDF (повільніше).",
        target, pdf_path.name,
    )
    try:
        import pdfplumber
    except ImportError:
        logger.warning("Пакет 'pdfplumber' не встановлено (pip install pdfplumber).")
        return None

    t_scan0 = time.monotonic()
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "".join((p.extract_text() or "") + "\n" for p in pdf.pages)
    except Exception as exc:
        logger.info("Не вдалось прочитати %s: %s", pdf_path, exc)
        return None
    finally:
        timing_logger.info(
            "[timing] pdf_full_scan file=%s number=§%s: %.3fs",
            pdf_path.name, target, time.monotonic() - t_scan0,
        )

    headers = list(_SECTION_HEADER_RE.finditer(text))
    for i, m in enumerate(headers):
        if m.group(1) == target:
            start = m.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            content = text[start:end].strip()
            return content[:max_chars] if content else None
    return None
