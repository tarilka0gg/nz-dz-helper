"""
Одноразове ручне налаштування Chromium-профілю для BrowserChatProvider
(solver.py). Відкриває ЗВИЧАЙНЕ (не headless) вікно з persistent-профілем —
залогинься вручну (включно з капчею/2FA, якщо буде), потім просто закрий
вікно браузера. Сесія (cookies/localStorage) залишиться в user_data_dir, і
BrowserChatProvider підхопить її автоматично при наступних запусках
solver.py — без повторного логіну.

Usage:
    python setup_browser_profile.py --service deepseek
    python setup_browser_profile.py --service gemini
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"  # src/ -> корінь проєкту


def _load_service_config(service: str) -> tuple[dict, dict]:
    """Повертає (browser_cfg, svc_cfg) — перший рівень потрібен для channel."""
    if not CONFIG_PATH.exists():
        print(f"Помилка: {CONFIG_PATH} не знайдено.", file=sys.stderr)
        sys.exit(2)

    with CONFIG_PATH.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    browser_cfg = config.get("browser_chat") or {}
    services = browser_cfg.get("services") or {}
    svc_cfg = services.get(service)
    if not svc_cfg:
        available = ", ".join(services) or "(жодного не налаштовано)"
        print(
            f"Помилка: немає config.yaml -> browser_chat.services.{service}. "
            f"Доступні: {available}.",
            file=sys.stderr,
        )
        sys.exit(2)
    return browser_cfg, svc_cfg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ручне (одноразове) логінення в чат для BrowserChatProvider."
    )
    parser.add_argument(
        "--service", required=True, help="Ключ з config.yaml -> browser_chat.services (deepseek/gemini)."
    )
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Помилка: пакет 'playwright' не встановлено "
            "(pip install playwright && playwright install chromium).",
            file=sys.stderr,
        )
        return 2

    browser_cfg, svc_cfg = _load_service_config(args.service)
    user_data_dir = Path(svc_cfg["user_data_dir"]).expanduser()
    url = svc_cfg["url"]
    channel = browser_cfg.get("channel")
    user_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Профіль: {user_data_dir}")
    print(f"Відкриваю {url} у видимому вікні {channel or 'Chromium'}...")
    print("Залогинься вручну (включно з капчею/2FA, якщо з'явиться),")
    print("потім просто закрий вікно браузера — сесія збережеться сама.")

    launch_kwargs: dict = {"headless": False}
    if channel:
        launch_kwargs["channel"] = channel

    with sync_playwright() as playwright:
        try:
            context = playwright.chromium.launch_persistent_context(
                str(user_data_dir), **launch_kwargs
            )
        except Exception as exc:
            print(
                f"Помилка запуску браузера (channel={channel or 'bundled chromium'}): {exc}",
                file=sys.stderr,
            )
            return 1
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(url)
        context.wait_for_event("close", timeout=0)

    print("Готово — сесію збережено в профілі.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
