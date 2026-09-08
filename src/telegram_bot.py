"""
Telegram-бот для розв'язувача ДЗ (nz_client.py + solver.py).

Команди:
  /today            — щоденник nz.ua на сьогодні, розв'язати кожне ДЗ
  /week             — щоденник на весь поточний тиждень
  /task <предмет> <номер> — розв'язати одне завдання вручну

Доступ обмежений allowlist'ом telegram user_id (TELEGRAM_ALLOWED_USER_IDS
в .env) — усі інші повідомлення ігноруються без жодної відповіді.

Запуск: python telegram_bot.py (polling, без публічного домену/webhook).
"""
from __future__ import annotations

import asyncio
import html as html_lib
import logging
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from telegram import Message, Update
from telegram.error import BadRequest, TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes

from nz_client import NzClient, NzLoginError, NzParseError
from solver import (
    LlmSolver,
    LlmSolverError,
    Task,
    _load_config,
    _tasks_from_diary,
    solve_task,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("telegram_bot")

NZ_COOKIES_PATH = Path(__file__).resolve().parent.parent / ".nz_cookies.pkl"  # src/ -> корінь проєкту
TELEGRAM_MESSAGE_LIMIT = 4000  # трохи запасу від реального ліміту 4096


def _load_allowed_user_ids() -> set[int]:
    raw = os.environ.get("TELEGRAM_ALLOWED_USER_IDS", "")
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError:
            logger.warning("TELEGRAM_ALLOWED_USER_IDS: не число проігноровано: %r", part)
    return ids


ALLOWED_USER_IDS = _load_allowed_user_ids()


def _allowed_only(handler):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if user is None or user.id not in ALLOWED_USER_IDS:
            logger.warning("Проігноровано повідомлення від чужого user_id=%s", user.id if user else None)
            return  # мовчки ігноруємо, як просив користувач
        await handler(update, context)

    return wrapper


# ---------------------------------------------------------------------- #
# Markdown (LLM-вивід, solver.py -> _EXPLAIN_SYSTEM/_ANSWER_SYSTEM) ->
# Telegram-HTML (parse_mode="HTML"). LLM більше не пише HTML-теги
# напряму — так надійніше (LLM значно рідше ламає прості **/_ пари, ніж
# збалансовані <b>/<i>), а вся відповідальність за коректний HTML — тут,
# в одному місці, а не розмазана по system prompt.
# ---------------------------------------------------------------------- #

_MD_CODE_BLOCK_RE = re.compile(r"```(?:\w+\n)?(.*?)```", re.DOTALL)
_MD_INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MD_BOLD_ALT_RE = re.compile(r"__(.+?)__", re.DOTALL)
_MD_ITALIC_STAR_RE = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")
_MD_ITALIC_UNDER_RE = re.compile(r"(?<![\w_])_(?!\s)([^_\n]+?)(?<!\s)_(?![\w_])")
_MD_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)")
_MD_HEADER_RE = re.compile(r"(?m)^#{1,6}[ \t]*(.+?)[ \t]*$")
_MD_BULLET_RE = re.compile(r"(?m)^([ \t]*)[-*•][ \t]+")


def markdown_to_telegram_html(text: str) -> str:
    """
    Простий конвертер Markdown -> Telegram-HTML. Порядок кроків важливий:

    1. Екранувати ВЕСЬ сирий текст одразу (html.escape) — щоб "<", ">", "&",
       які LLM написала як звичайний текст (не розмітку), не зламали
       HTML-парсинг Telegram. Markdown-символи (*, _, `, [, ], (, )) escape
       не займає, тож регулярки нижче однаково спрацюють на екранованому
       тексті.
    2. Код-блоки/inline-код виносяться в тимчасові плейсхолдери (\\x00N\\x00)
       ДО bold/italic — інакше "*"/"_" всередині коду (напр. формула з "_")
       могли б випадково зачепитись подальшими регулярками.
    3. bold -> <b>, italic -> <i>, посилання -> <a href>.
    4. Заголовки/маркери списків — БЕЗ тегів (Telegram parse_mode=HTML не
       підтримує <ul>/<li>/<h1..6> — рендерить як сирий текст): "### x"
       стає жирним рядком, "-"/"*" на початку рядка — звичайним "•".
    5. Плейсхолдери коду повертаються назад останніми.
    """
    escaped = html_lib.escape(text, quote=True)

    protected: list[str] = []

    def _protect(fragment: str) -> str:
        protected.append(fragment)
        return f"\x00{len(protected) - 1}\x00"

    escaped = _MD_CODE_BLOCK_RE.sub(lambda m: _protect(f"<pre>{m.group(1)}</pre>"), escaped)
    escaped = _MD_INLINE_CODE_RE.sub(lambda m: _protect(f"<code>{m.group(1)}</code>"), escaped)

    escaped = _MD_BOLD_RE.sub(lambda m: f"<b>{m.group(1)}</b>", escaped)
    escaped = _MD_BOLD_ALT_RE.sub(lambda m: f"<b>{m.group(1)}</b>", escaped)
    escaped = _MD_ITALIC_STAR_RE.sub(lambda m: f"<i>{m.group(1)}</i>", escaped)
    escaped = _MD_ITALIC_UNDER_RE.sub(lambda m: f"<i>{m.group(1)}</i>", escaped)
    escaped = _MD_LINK_RE.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', escaped)
    escaped = _MD_HEADER_RE.sub(lambda m: f"<b>{m.group(1)}</b>", escaped)
    escaped = _MD_BULLET_RE.sub(lambda m: f"{m.group(1)}• ", escaped)

    for idx, fragment in enumerate(protected):
        escaped = escaped.replace(f"\x00{idx}\x00", fragment)

    return escaped


_TAG_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)(?:\s[^>]*)?>")
_TAG_OR_TEXT_RE = re.compile(r"(<[^>]+>)")
_ALLOWED_TAGS = {"b", "i", "u", "s", "code", "pre", "a"}


def _strip_html_tags(html_text: str) -> str:
    """Фолбек, коли навіть коректний HTML чомусь не проходить parse_mode="HTML":
    прибирає теги, а НЕ показує їх користувачу сирими."""
    return _TAG_RE.sub("", html_text)


def split_html_safe(html_text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """
    Розбиває Telegram-HTML на шматки ≤limit, ніколи не розриваючи тег
    посередині (кожен <...> — атомарний токен, переноситься цілком) і
    завжди лишаючи теги збалансованими в кожному шматку: якщо розріз
    доводиться робити всередині відкритого <b>/<i>/... — тег закривається
    в кінці поточного шматка і відкривається знову на початку наступного
    (інакше форматування зникло б з одного боку розриву — саме той баг,
    що й малось на увазі: текст обривається між <b> і </b>).
    """
    if len(html_text) <= limit:
        return [html_text]

    tokens = [t for t in _TAG_OR_TEXT_RE.split(html_text) if t]
    chunks: list[str] = []
    current = ""
    open_stack: list[tuple[str, str]] = []  # (ім'я тега, повний відкриваючий тег)

    def close_suffix() -> str:
        return "".join(f"</{name}>" for name, _ in reversed(open_stack))

    def open_prefix() -> str:
        return "".join(tag_text for _, tag_text in open_stack)

    def flush() -> None:
        nonlocal current
        current += close_suffix()
        chunks.append(current)
        current = open_prefix()

    for token in tokens:
        m = _TAG_RE.fullmatch(token)
        if m:
            closing, name = m.group(1), m.group(2).lower()
            # Рахуємо суфікс закриття, який знадобиться ПІСЛЯ обробки цього
            # токена (не поточний!) — інакше при відкритті тега можна
            # вписати "<b>" впритул до ліміту, не лишивши місця навіть для
            # його власного "</b>", і отримати порожню пару "<b></b>" ЗА
            # межею ліміту при фінальному flush().
            if closing and open_stack and open_stack[-1][0] == name:
                prospective_stack = open_stack[:-1]
            elif closing:
                prospective_stack = open_stack
            else:
                prospective_stack = open_stack + [(name, token)]
            prospective_suffix = "".join(f"</{n}>" for n, _ in reversed(prospective_stack))

            if len(current) + len(token) + len(prospective_suffix) > limit:
                flush()
            current += token
            open_stack = prospective_stack
            continue

        remaining = token
        while remaining:
            available = limit - len(current) - len(close_suffix())
            if available <= 0:
                flush()
                continue
            if len(remaining) <= available:
                current += remaining
                remaining = ""
                break
            cut = max(remaining.rfind("\n", 0, available), remaining.rfind(" ", 0, available))
            if cut <= 0:
                cut = available
            current += remaining[:cut]
            remaining = remaining[cut:].lstrip("\n ")
            flush()

    if current.strip() and current != open_prefix():
        current += close_suffix()
        chunks.append(current)

    return chunks


async def _send_html(update: Update, html_text: str) -> None:
    """
    html_text — уже готовий Telegram-HTML (після markdown_to_telegram_html),
    зі збалансованими тегами з дозволеного набору. Якщо Telegram все ж
    поверне BadRequest (напр. непередбачена розмітка) — фолбек шле той
    самий шматок БЕЗ тегів (_strip_html_tags), а не з сирими "<b>" у тексті.
    """
    for chunk in split_html_safe(html_text):
        try:
            await update.message.reply_text(chunk, parse_mode="HTML")
        except BadRequest as exc:
            logger.warning("HTML parse_mode не спрацював (%s) — шлю без тегів.", exc)
            await update.message.reply_text(_strip_html_tags(chunk), parse_mode="HTML")


async def _send_reports(update: Update, reports: list[tuple[str, Optional[str]]]) -> None:
    """
    Кожен звіт (один предмет) — окреме повідомлення, а не один спільний текст.
    
    Fix #3: якщо reports містить image_url, шлемо фото ПЕРЕД текстом.
    """
    for report, image_url in reports:
        if image_url:
            try:
                await update.message.reply_photo(photo=image_url)
            except Exception as exc:
                logger.warning("Не вдалось відправити фото з ГДЗ: %s", exc)
        await _send_html(update, report)


async def _delete_all_after(messages: list[Message], delay: float) -> None:
    """Фонова задача (asyncio.create_task, не await у виклику) — чекає
    delay секунд і видаляє ВСІ messages. Telegram інколи відмовляє видаляти
    (повідомлення старше 48г, вже видалене іншим шляхом тощо) — це не
    помилка застосунку, тому кожне видалення ловиться окремо (одна невдача
    не має заважати видалити решту), лише логуємо, не валимось."""
    await asyncio.sleep(delay)
    for message in messages:
        try:
            await message.delete()
        except TelegramError as exc:
            logger.info("Не вдалось видалити повідомлення %s: %s", message.message_id, exc)


async def _status(update: Update, text: str) -> Message:
    """
    Короткі службові повідомлення бота ("Дивлюсь щоденник…", помилки
    nz.ua тощо). Завжди parse_mode="HTML" — без винятку для жодного
    send_message в цьому файлі — і завжди екранований текст: навіть
    сюди інколи потрапляють дані ззовні (текст винятку nz.ua, аргументи
    /task від користувача), які самі можуть містити "<"/">"/"&".
    split_html_safe тут теж рахує (nz.ua-помилка теоретично може бути
    довгою) — на екранованому тексті без тегів вона працює як звичайний
    розрізувач по рядках/словах.

    Повертає ОСТАННЄ надіслане повідомлення (для короткого статусного
    тексту тут завжди рівно один шматок) — виклики, яким треба потім
    видалити своє статусне повідомлення (_delete_all_after), збирають
    ці Message самі.
    """
    escaped = html_lib.escape(text, quote=True)
    msg = None
    for chunk in split_html_safe(escaped):
        msg = await update.message.reply_text(chunk, parse_mode="HTML")
    return msg


# ---------------------------------------------------------------------- #
# Спільний стан (лінива ініціалізація, з блокуванням проти паралельних
# команд, що ділять один requests.Session/LlmSolver-кеш провайдерів).
# ---------------------------------------------------------------------- #

_state_lock = asyncio.Lock()
_nz_client: Optional[NzClient] = None
_llm_solver: Optional[LlmSolver] = None
_config: Optional[dict] = None


def _get_config() -> dict:
    global _config
    if _config is None:
        _config = _load_config()
    return _config


def _ensure_nz_client() -> NzClient:
    """Синхронна частина — виконується в окремому потоці через to_thread."""
    global _nz_client
    if _nz_client is not None and _nz_client.is_logged_in():
        return _nz_client

    client = NzClient()
    if not (client.load_cookies(NZ_COOKIES_PATH) and client.is_logged_in()):
        username = os.environ["NZ_USERNAME"]
        password = os.environ["NZ_PASSWORD"]
        client.login(username, password)
        client.save_cookies(NZ_COOKIES_PATH)

    _nz_client = client
    return client


def _get_llm_solver() -> LlmSolver:
    global _llm_solver
    if _llm_solver is None:
        _llm_solver = LlmSolver(config=_get_config())
    return _llm_solver


def _format_result(subject: str, homework_text: str, result: dict) -> tuple[str, Optional[str]]:
    """
    LLM пише Markdown (solver.py _EXPLAIN_SYSTEM/_ANSWER_SYSTEM) — тут
    конвертуємо його в Telegram-HTML через markdown_to_telegram_html()
    (сам конвертер вже екранує "<"/">"/"&" і безпечно перетворює **/_/`
    на теги). subject/homework_text — сирий текст із щоденника, окремо
    екранується як завжди.
    
    Fix #3: повертає tuple (text, image_url), де image_url — URL скану ГДЗ,
    якщо його знайдено і він пройшов sanity-check (source містить "gdz").
    """
    hw_preview = homework_text if len(homework_text) <= 200 else homework_text[:200] + "…"
    answer_html = markdown_to_telegram_html(result["answer"])
    text = (
        f"📘 <b>{html_lib.escape(subject)}</b>\n"
        f"ДЗ: {html_lib.escape(hw_preview)}\n\n"
        f"{answer_html}\n\n"
        f"<i>[джерело: {html_lib.escape(result['source'])}, "
        f"confidence: {html_lib.escape(result['confidence'])}]</i>"
    )
    image_url = result.get("source_image_url") if "gdz" in result.get("source", "") else None
    return (text, image_url)


def _solve_tasks_blocking(tasks: list[Task]) -> list[tuple[str, Optional[str]]]:
    """
    Синхронна частина (мережеві виклики LLM/ГДЗ) — в окремому потоці.
    source="skipped" (просте "повторити" без контрольної, solver.py ->
    is_review_only) повністю ігнорується — жодної згадки в чаті, навіть
    не збирається в підсумок.
    
    Fix #3: повертає list[tuple[text, image_url]], де image_url — URL скану
    ГДЗ, якщо його знайдено і він пройшов sanity-check (source містить "gdz").

    LlmSolverError (напр. "усі провайдери недоступні" з LlmSolver.solve()'а
    після вичерпаного fallback-ланцюжка) сюди прилітає ВЖЕ як коротке
    людське повідомлення — solver.py навмисно ніколи не кладе туди
    сирий traceback/JSON, тож str(exc) безпечно показувати як є.
    """
    llm = _get_llm_solver()
    config = _get_config()
    reports: list[tuple[str, Optional[str]]] = []
    for task in tasks:
        try:
            result = solve_task(task, mode="explain", llm=llm, config=config)
        except LlmSolverError as exc:
            logger.exception(
                "Помилка розв'язку '%s' (%s)", task.subject, task.homework_text
            )
            reports.append(
                (
                    f"📘 <b>{html_lib.escape(task.subject)}</b>\n"
                    f"ДЗ: {html_lib.escape(task.homework_text)}\n"
                    f"{html_lib.escape(str(exc))}",
                    None,
                )
            )
            continue
        if result["source"] == "skipped":
            continue
        reports.append(_format_result(task.subject, task.homework_text, result))
    return reports


def _monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _find_today_day(days: list[dict]) -> Optional[dict]:
    for day in days:
        if day.get("day_label") and "сьогодні" in day["day_label"]:
            return day
    idx = date.today().weekday()
    return days[idx] if 0 <= idx < len(days) else None


def _fetch_diary_blocking(scope: str) -> list[dict]:
    """scope: 'today' | 'week'. Синхронна частина — в окремому потоці."""
    client = _ensure_nz_client()
    student_id = os.environ["NZ_STUDENT_ID"]
    monday = _monday_of(date.today()).isoformat()
    days = client.get_diary_week(monday, student_id)

    if scope == "today":
        today_day = _find_today_day(days)
        return [today_day] if today_day else []
    return days


# ---------------------------------------------------------------------- #
# Команди
# ---------------------------------------------------------------------- #

@_allowed_only
async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status_messages = [await _status(update, "Дивлюсь щоденник на сьогодні…")]
    async with _state_lock:
        try:
            days = await asyncio.to_thread(_fetch_diary_blocking, "today")
        except (NzLoginError, NzParseError) as exc:
            await _status(update, f"Помилка nz.ua: {exc}")
            return

        tasks = _tasks_from_diary(days)
        if not tasks:
            await _status(update, "На сьогодні ДЗ не знайдено (або уроків немає).")
            return

        status_messages.append(await _status(update, f"Знайшов {len(tasks)} завдань, розв'язую…"))
        # Обидва статусні повідомлення видаляються разом через 2.5с після
        # другого з них — саме тоді розв'язування вже почалось, і вони
        # свою роль ("я живий, працюю") виконали.
        asyncio.create_task(_delete_all_after(status_messages, 2.5))
        reports = await asyncio.to_thread(_solve_tasks_blocking, tasks)

    await _send_reports(update, reports)


@_allowed_only
async def cmd_week(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status_messages = [await _status(update, "Дивлюсь щоденник на весь тиждень…")]
    async with _state_lock:
        try:
            days = await asyncio.to_thread(_fetch_diary_blocking, "week")
        except (NzLoginError, NzParseError) as exc:
            await _status(update, f"Помилка nz.ua: {exc}")
            return

        tasks = _tasks_from_diary(days)
        if not tasks:
            await _status(update, "На цьому тижні ДЗ не знайдено.")
            return

        status_messages.append(
            await _status(update, f"Знайшов {len(tasks)} завдань, розв'язую… (це займе трохи часу)")
        )
        # Обидва статусні повідомлення видаляються разом через 2.5с після
        # другого з них — саме тоді розв'язування вже почалось.
        asyncio.create_task(_delete_all_after(status_messages, 2.5))
        reports = await asyncio.to_thread(_solve_tasks_blocking, tasks)

    await _send_reports(update, reports)


@_allowed_only
async def cmd_task(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args
    if len(args) < 2:
        await _status(update, "Формат: /task <предмет> <номер>, напр. /task Алгебра 1.13")
        return

    number = args[-1]
    subject = " ".join(args[:-1])
    task = Task(subject=subject, homework_text=f"№{number}")

    await _status(update, f"Розв'язую {subject} №{number}…")
    async with _state_lock:
        reports = await asyncio.to_thread(_solve_tasks_blocking, [task])

    if reports:
        report, image_url = reports[0]
        if image_url:
            try:
                await update.message.reply_photo(photo=image_url)
            except Exception as exc:
                logger.warning("Не вдалось відправити фото з ГДЗ: %s", exc)
        await _send_html(update, report)


@_allowed_only
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _status(
        update,
        "Бот-розв'язувач ДЗ nz.ua.\n\n"
        "/today — ДЗ на сьогодні\n"
        "/week — ДЗ на весь тиждень\n"
        "/task <предмет> <номер> — розв'язати вручну (напр. /task Алгебра 1.13)",
    )


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Помилка: TELEGRAM_BOT_TOKEN не задано в .env")
        return 2
    if not ALLOWED_USER_IDS:
        print("Помилка: TELEGRAM_ALLOWED_USER_IDS не задано в .env (порожній allowlist)")
        return 2

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("week", cmd_week))
    app.add_handler(CommandHandler("task", cmd_task))

    logger.info("Бот запущено (allowlist: %s)", ALLOWED_USER_IDS)
    app.run_polling()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
