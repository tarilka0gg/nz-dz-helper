# nz-dz-helper

Telegram-бот, який тягне щоденник дитини з nz.ua (Yii2 SSR-портал "Нові
знання"), знаходить домашні завдання і допомагає з ними: для вправ з
підручника — знаходить умову (PDF підручника, опційно ГДЗ-скан + OCR) і
дає пояснення через LLM; для творів/есе — пише готовий текст; організаційні
завдання ("повторити", "вивчити терміни") — коротка порада або взагалі
пропускаються, якщо не потребують розбору.

## Структура проєкту

```
nz-dz-helper/
├── src/
│   ├── nz_client.py           # логін + парсинг щоденника nz.ua
│   ├── solver.py               # класифікація ДЗ, LLM-пайплайн, GDZ/OCR/PDF
│   ├── textbook_source.py      # завантаження PDF підручника, витяг умови
│   ├── telegram_bot.py         # Telegram-бот (команди /today /week /task)
│   └── setup_browser_profile.py  # опційний ручний логін для BrowserChatProvider
├── config.yaml                 # предмети, LLM-провайдери, ключові слова, підручники
├── .env.example                 # шаблон змінних середовища (скопіювати в .env)
├── requirements.txt
├── start.sh                     # активує/створює venv і запускає бота
└── .gitignore
```

Runtime-дані (генеруються самі при першому запуску, у git не потрапляють):
`.env`, `.nz_cookies.pkl`, `nz_solver.db` (SQLite OCR-кеш), `.textbook_cache/`
(кешовані PDF), `*.log`.

## Встановлення

```bash
git clone <шлях-до-цього-репозиторію>   # або просто перейти в папку, якщо вже тут
cd nz-dz-helper
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # і заповнити реальними значеннями
```

Або простіше — одразу через `start.sh` (сам створить venv і поставить
залежності, якщо їх ще нема):

```bash
./start.sh
```

### Змінні середовища (`.env`)

| Змінна | Що це |
|---|---|
| `NZ_USERNAME`, `NZ_PASSWORD` | логін/пароль на nz.ua |
| `NZ_STUDENT_ID` | ID учня в щоденнику nz.ua |
| `GROQ_API_KEY` | безкоштовний ключ з console.groq.com |
| `GEMINI_API_KEY` | безкоштовний ключ з aistudio.google.com |
| `TELEGRAM_BOT_TOKEN` | токен бота від **@BotFather** (`/newbot` в Telegram) |
| `TELEGRAM_ALLOWED_USER_IDS` | твій Telegram user_id (дізнатись у **@userinfobot**), через кому якщо декілька |

## Запуск

```bash
./start.sh
```

(або вручну: `.venv/bin/python src/telegram_bot.py` — обов'язково з кореня
проєкту, бо `config.yaml`/`.env`/кеші шукаються відносно поточної теки).

### Команди бота

- `/today` — щоденник на сьогодні, розв'язати кожне ДЗ
- `/week` — те саме на весь тиждень
- `/task <предмет> <номер>` — розв'язати одне завдання вручну (напр. `/task Алгебра 1.13`)

Доступ обмежений allowlist'ом `TELEGRAM_ALLOWED_USER_IDS` — усі інші
повідомлення ігноруються без відповіді.

## LLM-провайдери

`config.yaml -> llm`: `default_provider`/`overrides` за предметом (Groq —
дефолт, Gemini — для точних наук). При rate-limit/quota (429) один
провайдер автоматично фолбекає на наступний з `llm.fallback_order`; якщо
впали всі — користувач бачить коротке "недоступно, спробуй пізніше",
без сирих деталей помилки.

### BrowserChatProvider (опційно, непрацездатно за замовчуванням)

`config.yaml -> browser_chat` — автоматизація веб-чату DeepSeek/Gemini AI
Studio через Playwright замість офіційного API. **Порушує ToS цих
сервісів** і на момент написання заблоковане антибот-захистом обох
сервісів — лишається задокументованим, не рекомендованим шляхом. Перед
використанням (якщо колись запрацює): `python src/setup_browser_profile.py --service deepseek`.

## CLI (без Telegram)

```bash
.venv/bin/python src/solver.py --diary diary_sample.json
```

`diary_sample.json` — не в репозиторії (реальні дані дитини), згенеруй
свій через `nz_client.py` або `/week` в боті й збережи вивід у файл.
