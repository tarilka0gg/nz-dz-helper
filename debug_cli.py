#!/usr/bin/env python3
"""
Debug-CLI для telegram_bot.py — викликає СПРАВЖНІ обробники команд бота
(cmd_today/cmd_week/cmd_task, ті самі функції, що й реальний Telegram-
polling) з підробленим Update/Context, замість реальної відправки в чат
друкує (і повністю логує в файл) усе, що бот відправив би: кожен текст
повідомлення, кожне фото (URL), кожне видалення статусного повідомлення.

Навіщо: щоб самостійно (без доступу до чужого Telegram-чату) прогнати
/week чи /today і подивитись РІВНО те, що виконає бот — той самий код,
той самий allowlist-декоратор (_allowed_only), той самий шлях розв'язку
завдань (_solve_and_send_tasks) — а не імітацію чи окремий тестовий
скрипт, що дублює логіку.

Використання:
    ./debug_cli.py week
    ./debug_cli.py today
    ./debug_cli.py task Алгебра 1.13
    ./debug_cli.py week --verbose      # DEBUG і в консоль, не лише у файл

Повне логування: усе (DEBUG, усі наші логери: telegram_bot, solver,
solver.timing, nz_client, textbook_source) завжди пишеться в
debug_cli.log поруч зі скриптом, незалежно від --verbose — той файл і є
"повне логування" для розбору після прогону.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LOG_FILE = PROJECT_ROOT / "debug_cli.log"


def _setup_logging(verbose: bool) -> None:
    """
    БЕЗ --verbose: файл на INFO — цього достатньо, щоб бачити кожен крок
    (subject/task, timing, помилки), і не вибухнути в розмір. Один живий
    /week на DEBUG дав 935 МБ / 9.3М рядків за ~3 хв — через
    Book4Source._find_matching_link/_text_matches (src/solver.py), які
    логують DEBUG на КОЖНЕ перевірене посилання під час ГДЗ-пошуку (не
    моя інструментація — лишилось від попередніх правок; не займався
    прибиранням, лише вимкнув тут за замовчуванням). --verbose вмикає
    DEBUG і для файлу, і для консолі — свідомо, коли справді треба
    заглибитись у конкретний випадок.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console_handler)

    # httpx/openai логи в DEBUG дуже шумні (тіло кожного запиту) — у файл
    # все одно пишемо (DEBUG на root), але явно приглушуємо їхній рівень,
    # щоб не забивали власне важливі DEBUG-рядки нашого коду.
    for noisy in ("httpx", "httpx2", "openai._base_client", "google_genai.models", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.INFO)


class FakeMessage:
    """Мінімальна заміна telegram.Message — лише те, що реально викликає
    telegram_bot.py (.reply_text, .reply_photo, .delete, .message_id)."""

    _id_counter = itertools.count(1)

    def __init__(self, text: str = ""):
        self.message_id = next(self._id_counter)
        self.text = text

    async def reply_text(self, text: str, parse_mode: str | None = None, **kwargs) -> "FakeMessage":
        print(f"\n--- 📨 БОТ НАДІСЛАВ ТЕКСТ (parse_mode={parse_mode}) ---")
        print(text)
        print("--- кінець повідомлення ---")
        logging.getLogger("debug_cli").info(
            "reply_text(parse_mode=%s): %s", parse_mode, text
        )
        return FakeMessage(text)

    async def reply_photo(self, photo, **kwargs) -> "FakeMessage":
        print(f"\n--- 🖼️ БОТ НАДІСЛАВ ФОТО ---\n{photo}\n--- кінець фото ---")
        logging.getLogger("debug_cli").info("reply_photo: %s", photo)
        return FakeMessage()

    async def delete(self) -> None:
        print(f"--- 🗑️ БОТ ВИДАЛИВ повідомлення #{self.message_id} ---")
        logging.getLogger("debug_cli").info("delete message_id=%s", self.message_id)


class FakeUser:
    def __init__(self, user_id: int):
        self.id = user_id


class FakeChat:
    def __init__(self, chat_id: int):
        self.id = chat_id


class FakeUpdate:
    def __init__(self, user_id: int):
        self.message = FakeMessage()
        self.effective_user = FakeUser(user_id)
        self.effective_chat = FakeChat(user_id)


class FakeContext:
    def __init__(self, args: list[str]):
        self.args = args


async def _run(command: str, args: list[str]) -> None:
    # Імпорт ПІСЛЯ sys.path.insert і ПІСЛЯ _setup_logging (нижче в main) —
    # telegram_bot.py читає ALLOWED_USER_IDS/токени з .env одразу при
    # імпорті модуля.
    import telegram_bot as bot

    if not bot.ALLOWED_USER_IDS:
        print(
            "Помилка: TELEGRAM_ALLOWED_USER_IDS порожній у .env — "
            "_allowed_only відхилить будь-який виклик.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Перший id з allowlist — той самий шлях, що й реальний дозволений
    # користувач, включно з _allowed_only-перевіркою (не обходимо її, а
    # проходимо як насправді).
    real_user_id = next(iter(bot.ALLOWED_USER_IDS))
    update = FakeUpdate(real_user_id)
    context = FakeContext(args)

    handlers = {
        "today": bot.cmd_today,
        "week": bot.cmd_week,
        "task": bot.cmd_task,
        "start": bot.cmd_start,
    }
    handler = handlers.get(command)
    if handler is None:
        print(f"Невідома команда: {command!r}. Доступні: {', '.join(handlers)}", file=sys.stderr)
        sys.exit(2)

    print(f"=== Викликаю /{command} (user_id={real_user_id}, args={args}) ===")
    await handler(update, context)
    print(f"\n=== /{command} завершено ===")


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug-CLI для telegram_bot.py — прогнати команду бота напряму.")
    parser.add_argument("command", choices=["today", "week", "task", "start"])
    parser.add_argument("task_args", nargs="*", help="Для /task: <предмет> <номер>, напр. Алгебра 1.13")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG-логи також у консоль (не лише у файл).")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    print(f"Повний DEBUG-лог пишеться в: {LOG_FILE}")

    asyncio.run(_run(args.command, args.task_args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
