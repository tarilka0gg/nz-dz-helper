"""
Витягування умов задач напряму з PDF підручника (pidruchnyk.com.ua тощо) —
альтернатива/доповнення до OCR зі сканів ГДЗ (solver.py). PDF зазвичай дає
чистіший текст (без плутанини на кшталт "2n" замість "2ⁿ", яку ловить OCR
зображень), тому це primary-джерело умови задачі для LlmSolver.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("textbook_source")

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".textbook_cache"  # src/ -> корінь проєкту

# Заголовок пункту підручника Мерзляка: "1.13." опційно з позначкою рівня
# складності (° / • / ••), на початку рядка/абзацу.
_EXERCISE_HEADER_RE = re.compile(r"(?m)^\s*(\d+\.\d+)\.\s*[°•]{0,2}\s*")


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
    return dest


def _extract_page_text(pdf, index: int) -> str:
    if 0 <= index < len(pdf.pages):
        return pdf.pages[index].extract_text() or ""
    return ""


def _scan_for_exercise(pdf, indices, target: str) -> Optional[str]:
    text = "".join(_extract_page_text(pdf, i) + "\n" for i in indices)
    headers = list(_EXERCISE_HEADER_RE.finditer(text))
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
    Шукає в PDF пункт з номером exercise_number (формат "1.13", як у
    Мерзляка) і повертає текст від цього номера до наступного такого ж
    заголовка. None, якщо не знайдено (сторінка/номер відсутні, PDF
    пошкоджений тощо — ніколи не кидає виняток назовні).

    Якщо задано page_hint (друкований номер сторінки підручника) — спершу
    звужуємо пошук до вузького вікна навколо неї (щоб не зловити однаковий
    номер з іншого розділу), і лише якщо там не знайшлось — скануємо весь
    документ.
    """
    target = str(exercise_number).strip()
    if not target:
        return None

    try:
        import pdfplumber
    except ImportError:
        logger.warning("Пакет 'pdfplumber' не встановлено (pip install pdfplumber).")
        return None

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
                    result = _scan_for_exercise(pdf, window, target)
                    if result:
                        return result
                except ValueError:
                    pass

            return _scan_for_exercise(pdf, range(total_pages), target)
    except Exception as exc:  # пошкоджений/нечитний PDF — не валимо виклик
        logger.info("Не вдалось прочитати %s: %s", pdf_path, exc)
        return None
