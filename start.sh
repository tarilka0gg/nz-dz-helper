#!/usr/bin/env bash
# Запускає telegram_bot.py: створює venv і ставить залежності, якщо їх
# ще нема, потім запускає бота. Завжди з кореня проєкту (config.yaml,
# .env, кеші шукаються відносно поточної теки) — тому спершу cd сюди.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
    echo "venv не знайдено — створюю .venv і ставлю залежності..."
    python3 -m venv .venv
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -r requirements.txt
fi

if [ ! -f ".env" ]; then
    echo "Помилка: .env не знайдено. Скопіюй .env.example в .env і заповни значення." >&2
    exit 1
fi

exec .venv/bin/python src/telegram_bot.py
