import os
import io
import time
import base64
import logging
import asyncio
import re
from typing import Tuple, Optional, Any, Dict
from datetime import time as tm, datetime, timezone
from urllib.parse import urlparse
import requests
from requests import RequestException
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
)
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    CallbackContext,
    CallbackQueryHandler,
    filters,
)
from dotenv import load_dotenv
load_dotenv()

# ----------------- ЛОГИ -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ----------------- НАСТРОЙКИ -----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "ВВЕДИТЕ СВОЙ ТОКЕН")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "ВВЕДИТЕ СВОЙ ТОКЕН")
FOLDER_ID = os.getenv("FOLDER_ID", "ВВЕДИТЕ СВОЙ ID")
CATALOG_ID = os.getenv("CATALOG_ID", "ВВЕДИТЕ СВОЙ ID")
VK_SERVICE_TOKEN = os.getenv(
    "VK_SERVICE_TOKEN",
    "ВВЕДИТЕ СВОЙ ТОКЕН",
)

# ----------------- ОПИСАНИЯ КНОПОК -----------------
BUTTON_DESCRIPTIONS = {
    "✨ Создать пост": (
        "✨ Здесь вы можете создать новый контент для соцсетей.\n"
        "Выберите: текстовый пост или изображение — и бот сгенерирует его за вас!"
    ),
    "✍️ Исправить текст": (
        "✍️ Отправьте любой текст, и бот исправит орфографию, пунктуацию и стиль, "
        "сделав его готовым к публикации."
    ),
    "📄 Пост": (
        "📤 Показывает последний сгенерированный вами пост (текст + изображение, если есть)."
    )
}

# ----------------- STATES -----------------
TEXT_INPUT, IMAGE_DESC, EDIT_TEXT, PLAN_DAYS, NAME, DESCRIPTION, ACTIVITY, AUDIENCE, LOCATION, CONTACT = range(10)

# ----------------- Фильтрация неподходящих тем -----------------
INAPPROPRIATE_TOPICS_KEYWORDS = {
    "политик", "религи", "наркотик", "оружи", "войн", "терроризм", "порнограф", "жестокост",
    "расизм", "экстремизм", "суицид", "ненависть",
    "пропаганда", "дискриминация", "насили", "порно", "интим", "эротик", "секс", "оргазм",
    "гомосексуализм", "проституция", "алкогол", "курение", "лгбт", "вульгарн",
    "оскорб", "клевет", "фейк", "дезинформация", "протест", "митинг", "восстани", "бунт",
}

def is_topic_inappropriate(idea: str) -> bool:
    idea_lower = idea.lower()
    return any(word in idea_lower for word in INAPPROPRIATE_TOPICS_KEYWORDS)

# ----------------- Утилиты -----------------
def has_org_data(chat_data: Dict) -> bool:
    keys = ("name", "description", "org_activity", "org_audience", "org_location", "org_contact")
    return any(chat_data.get(k) not in (None, "", "(не указано)") for k in keys)

def main_menu_keyboard(chat_data: Optional[Dict] = None):
    status = " ✅" if chat_data and has_org_data(chat_data) else ""
    kb = [
        [InlineKeyboardButton(f"✨ Создать пост{status}", callback_data="menu_generate")],
        [InlineKeyboardButton("✍️ Исправить текст", callback_data="menu_editing")],
        [InlineKeyboardButton("🗓️ Контент-план", callback_data="menu_plan")],
        [InlineKeyboardButton("🏢 О НКО", callback_data="menu_data")],
        [InlineKeyboardButton("ℹ️ Как пользоваться?", callback_data="menu_help")],
    ]
    return InlineKeyboardMarkup(kb)

def back_button(back_callback: str):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("↩️ Назад", callback_data=back_callback)]
    ])

def regenerate_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("↩️ Назад", callback_data="main_menu"),
            InlineKeyboardButton("🔁 Другой вариант", callback_data="regenerate")
        ]
    ])

def generate_sub_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✍️ Текст (сам напишу)", callback_data="menu_text")],
        [InlineKeyboardButton("🖼️ Картинку к посту", callback_data="menu_image")],
        [InlineKeyboardButton("↩️ Назад", callback_data="main_menu")]
    ])

def generate_post_options_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✍️ Текст (сам напишу)", callback_data="menu_text")],
        [InlineKeyboardButton("🖼️ Картинку к посту", callback_data="menu_image")],
        [InlineKeyboardButton("↩️ Назад", callback_data="main_menu")]
    ])

def editing_sub_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✍️ Исправить текст", callback_data="menu_edit")],
        [InlineKeyboardButton("↩️ Назад", callback_data="main_menu")]
    ])

def plan_sub_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🗓️ На 7 дней", callback_data="plan_7")],
        [InlineKeyboardButton("📆 На 30 дней", callback_data="plan_30")],
        [InlineKeyboardButton("🔢 Указать вручную", callback_data="menu_plan_custom")],
        [InlineKeyboardButton("↩️ Назад", callback_data="main_menu")]
    ])

def data_sub_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🖊️ Заполнить данные", callback_data="menu_info")],
        [InlineKeyboardButton("👀 Показать данные", callback_data="menu_show_info")],
        [InlineKeyboardButton("🗑️ Удалить данные НКО", callback_data="menu_clear_info")],
        [InlineKeyboardButton("↩️ Назад", callback_data="main_menu")]
    ])

def plan_result_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("↩️ Назад", callback_data="main_menu")],
        [InlineKeyboardButton("🔔 Включить оповещение", callback_data="fake_reminder")]
    ])

async def send_chat_action(context: CallbackContext, chat_id: int, action: ChatAction = ChatAction.TYPING):
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=action)
    except Exception as e:
        logger.debug("send_chat_action failed: %s", e)

def requests_with_retry(method: str, url: str, retries: int = 3, backoff: float = 0.6, **kwargs) -> requests.Response:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.request(method, url, timeout=40, **kwargs)
            resp.raise_for_status()
            return resp
        except RequestException as e:
            last_exc = e
            logger.warning("HTTP %s %s failed attempt %d/%d: %s", method, url, attempt, retries, e)
            time.sleep(backoff * attempt)
    raise last_exc

# ----------------- VK utilities -----------------
def extract_vk_domain(url: str) -> Optional[str]:
    try:
        if not url:
            return None
        url = url.strip()
        if not url.startswith("http"):
            url = "https://" + url
        p = urlparse(url)
        if "vk.com" not in p.netloc:
            return None
        path = p.path.strip("/")
        if not path:
            return None
        domain = path.split("/")[0]
        domain = domain.split("?")[0]
        return domain
    except Exception:
        return None

def fetch_vk_posts(domain: str, token: str, count: int = 5) -> Tuple[Optional[list], Optional[str]]:
    if not token:
        return None, "VK token не предоставлен"
    try:
        url = "https://api.vk.com/method/wall.get"
        params = {
            "domain": domain,
            "count": count,
            "access_token": token,
            "v": "5.131",
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        j = resp.json()
        if "error" in j:
            err = j["error"]
            return None, f"VK API error {err.get('error_code')}: {err.get('error_msg')}"
        response = j.get("response", {})
        items = response.get("items", [])
        posts = []
        for it in items:
            text = it.get("text", "") or ""
            posts.append({"id": it.get("id"), "text": text})
        return posts, None
    except Exception as e:
        return None, f"VK fetch error: {e}"

# ----------------- YANDEX: ТЕКСТ -----------------
async def yandex_generate_text(prompt_text: str) -> Tuple[Optional[str], Optional[str]]:
    if not YANDEX_API_KEY:
        return None, "YANDEX_API_KEY не установлен!"
    if FOLDER_ID:
        model_uri = f"gpt://{FOLDER_ID}/yandexgpt-lite/latest"
    else:
        model_uri = "yandex/gpt-lite"
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    payload = {
        "modelUri": model_uri,
        "completionOptions": {"temperature": 0.6, "maxTokens": 400},
        "messages": [{"role": "user", "text": prompt_text}],
    }
    headers = {"Authorization": f"Api-Key {YANDEX_API_KEY}", "Content-Type": "application/json"}
    try:
        resp = await asyncio.to_thread(requests_with_retry, "POST", url, json=payload, headers=headers)
        try:
            j = resp.json()
        except ValueError:
            logger.error("Yandex returned invalid JSON: %s", resp.text)
            return None, f"Невалидный JSON в ответе: {resp.text}"
        if isinstance(j, dict) and j.get("error"):
            err = j["error"]
            http_code = err.get("httpCode") or resp.status_code
            msg = err.get("message") or str(err)
            logger.error("Yandex API error %s: %s", http_code, msg)
            return None, f"Yandex API error {http_code}: {msg}"
        def extract_text(obj: Any) -> Optional[str]:
            if obj is None:
                return None
            if isinstance(obj, str):
                return obj.strip()
            if isinstance(obj, dict):
                for k in ("text", "content", "message", "output", "result"):
                    if k in obj and isinstance(obj[k], (str, dict, list)):
                        found = extract_text(obj[k])
                        if found:
                            return found
                for k in ("alternatives", "outputs", "choices"):
                    alts = obj.get(k)
                    if isinstance(alts, list) and len(alts) > 0:
                        first = alts[0]
                        if isinstance(first, dict):
                            if "message" in first:
                                return extract_text(first["message"])
                            for kk in ("text", "content"):
                                if kk in first:
                                    return extract_text(first[kk])
                        else:
                            return extract_text(first)
            if isinstance(obj, list):
                for el in obj:
                    found = extract_text(el)
                    if found:
                        return found
            return None
        text = extract_text(j.get("result", j))
        if not text:
            text = j.get("text") or j.get("content")
        if not text:
            logger.error("Не удалось распарсить текст из ответа Yandex: %s", j)
            return None, f"Не удалось распарсить текст из ответа: {j}"
        text = text.strip()
        if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
            text = text[1:-1].strip()
        return text, None
    except Exception as e:
        logger.exception("Ошибка при вызове Yandex text API: %s", e)
        return None, f"HTTP error: {e}"

# ----------------- YANDEX: ИЗОБРАЖЕНИЯ -----------------
async def yandex_generate_image(prompt: str, max_poll_seconds: int = 60) -> Tuple[Optional[bytes], Optional[str]]:
    if not YANDEX_API_KEY:
        return None, "YANDEX_API_KEY не установлен!"
    headers = {"Authorization": f"Api-Key {YANDEX_API_KEY}", "Content-Type": "application/json"}
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/imageGenerationAsync"
    if CATALOG_ID:
        model_uri = f"art://{CATALOG_ID}/yandex-art/latest"
    else:
        model_uri = "yandex/art"
    payload = {
        "modelUri": model_uri,
        "messages": [{"text": prompt}],
        "generationOptions": {"seed": "0"},
    }
    try:
        resp_post = await asyncio.to_thread(requests_with_retry, "POST", url, json=payload, headers=headers)
        try:
            data = resp_post.json()
        except ValueError:
            logger.error("Yandex image POST returned invalid JSON: %s", resp_post.text)
            return None, f"Ошибка: сервер вернул не JSON: {resp_post.text}"
        if not isinstance(data, dict):
            logger.error("Yandex image POST unexpected type: %s", type(data))
            return None, f"Некорректный формат ответа POST: {data}"
        if data.get("error"):
            msg = data["error"].get("message") if isinstance(data["error"], dict) else str(data["error"])
            logger.error("Yandex image API error: %s", msg)
            return None, f"Yandex error: {msg}"
        op_id = data.get("id") or data.get("operation_id") or data.get("operationId")
        if not op_id:
            logger.error("Yandex image POST did not return operation id: %s", data)
            return None, f"Ошибка: API не вернул ID операции. Ответ: {data}"
        poll_url = f"https://operation.api.cloud.yandex.net/operations/{op_id}"
        start_time = time.time()
        while True:
            if time.time() - start_time > max_poll_seconds:
                logger.error("Image generation timeout (op_id=%s)", op_id)
                return None, "Таймаут генерации изображения."
            await asyncio.sleep(1.5)
            try:
                resp_poll = await asyncio.to_thread(requests_with_retry, "GET", poll_url, headers=headers)
            except Exception as e:
                logger.warning("Polling failed: %s", e)
                continue
            raw = resp_poll.text
            logger.debug("POLL RAW: %s", raw)
            try:
                res = resp_poll.json()
            except ValueError:
                logger.error("Polling returned non-JSON: %s", raw)
                continue
            if not isinstance(res, dict):
                logger.error("Polling returned not-dict: %s", res)
                continue
            done = res.get("done") or False
            if not done:
                continue
            response = res.get("response") or res.get("result") or res.get("outputs")
            if isinstance(response, str):
                logger.error("Image op ended with message: %s", response)
                return None, f"Ошибка: операция завершилась с сообщением: {response}"
            if not isinstance(response, (dict, list)):
                logger.error("Unexpected response format for image op: %s", response)
                return None, f"Некорректный формат response: {response}"
            def extract_b64(obj):
                if isinstance(obj, dict):
                    for v in obj.values():
                        f = extract_b64(v)
                        if f:
                            return f
                elif isinstance(obj, list):
                    for v in obj:
                        f = extract_b64(v)
                        if f:
                            return f
                elif isinstance(obj, str):
                    if len(obj) > 200 and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\r\n" for c in obj.strip()):
                        return obj
                return None
            img_b64 = extract_b64(response)
            if not img_b64:
                logger.error("Image data not found in response: %s", response)
                return None, f"Изображение не найдено в response: {response}"
            try:
                img_bytes = base64.b64decode(img_b64)
            except Exception as e:
                logger.exception("Base64 decode error: %s", e)
                return None, "Ошибка декодирования base64"
            return img_bytes, None
    except Exception as e:
        logger.exception("Ошибка в yandex_generate_image: %s", e)
        return None, f"Ошибка генерации изображения: {e}"

# ----------------- Напоминание (заглушка) -----------------
async def fake_reminder_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()
    try:
        await query.message.delete()
    except:
        pass
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Оповещения будут приходить каждый день в 10:00",
        reply_markup=back_button("main_menu")
    )

# ----------------- Хэлперы -----------------
async def show_main_menu(update: Update, context: CallbackContext):
    chat_data = context.chat_data
    text = "Привет! Я помогу тебе с контентом для НКО 🙌\n"
    if has_org_data(chat_data):
        text += "✅ Данные НКО сохранены\n"
    text += "Начни с выбора действия ниже - или введи /help, чтобы увидеть все возможности."
    reply_markup = main_menu_keyboard(chat_data)
    chat_id = update.effective_chat.id
    if update.callback_query:
        try:
            await update.callback_query.message.delete()
        except:
            pass
        await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)
    elif update.message:
        await update.message.reply_text(text=text, reply_markup=reply_markup)

# ----------------- Команды и обработчики -----------------
async def start(update: Update, context: CallbackContext):
    context.user_data.clear()
    await show_main_menu(update, context)
    reply_keyboard = [["📄Посмотреть готовый пост"]]
    await update.message.reply_text(
        "Посмотреть свой пост ты сможешь нажав кнопку ниже\n",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard,
            resize_keyboard=True,
            one_time_keyboard=False
        )
    )
    return ConversationHandler.END

async def menu_command(update: Update, context: CallbackContext):
    await show_main_menu(update, context)

async def help_command(update: Update, context: CallbackContext):
    help_text = (
        "Доступные команды:\n"
        "/menu - возвращает в меню\n"
        "/text - сгенерировать текст поста\n"
        "/image - сгенерировать изображение\n"
        "/edit - исправить и улучшить текст\n"
        "/plan - составить контент-план (на N дней)\n"
        "/orginfo - заполнить или показать данные НКО\n"
        "💡 Совет: сначала заполните данные НКО — так контент будет точнее под вашу цель."
    )
    if update.message:
        await update.message.reply_text(help_text)
    elif update.callback_query:
        try:
            await update.callback_query.message.delete()
        except:
            pass
        await context.bot.send_message(chat_id=update.effective_chat.id, text=help_text)

# ----------------- Команды: entrypoints -----------------
async def text_command(update: Update, context: CallbackContext):
    context.user_data["expect"] = "text"
    await update.message.reply_text("Введите идею поста:", reply_markup=back_button("menu_generate"))
    return TEXT_INPUT

async def image_command(update: Update, context: CallbackContext):
    context.user_data["expect"] = "image"
    await update.message.reply_text("Опишите изображение:", reply_markup=back_button("menu_generate"))
    return IMAGE_DESC

async def edit_command(update: Update, context: CallbackContext):
    context.user_data["expect"] = "edit"
    await update.message.reply_text("Отправьте текст, который нужно исправить:", reply_markup=back_button("menu_editing"))
    return EDIT_TEXT

async def plan_command(update: Update, context: CallbackContext):
    context.user_data["expect"] = "plan"
    await update.message.reply_text("Введите количество дней (число):", reply_markup=back_button("menu_plan"))
    return PLAN_DAYS

async def orginfo_command(update: Update, context: CallbackContext):
    context.user_data["expect"] = "org_name"
    await update.message.reply_text(
        "Давайте введём данные о вашей НКО. Вы можете в любой момент отправить «Пропустить», чтобы не указывать поле.\n"
        "1/6. Введите название организации (или напиши текстом «Пропустить»):",
        reply_markup=back_button("menu_data"),
    )
    return NAME

# ----------------- VK: подтверждение -----------------
async def vk_confirm_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id
    try:
        await query.message.delete()
    except:
        pass

    if query.data == "vk_confirm_yes":
        domain = context.user_data.get("vk_pending_domain")
        if not domain:
            await context.bot.send_message(chat_id=chat_id, text="Ошибка: нет данных VK.", reply_markup=main_menu_keyboard(context.chat_data))
            return
        posts, err = fetch_vk_posts(domain, VK_SERVICE_TOKEN, count=5)
        if err:
            logger.warning("VK fetch error: %s", err)
            context.chat_data["vk_summary"] = ""
        else:
            texts = [p["text"] for p in posts if p.get("text")]
            if texts:
                sample = "\n---\n".join(texts[:3])
                summary_prompt = f"На основе этих последних постов из VK опиши общий стиль, тон и формат постов (без цитирования):\n{sample}"
                summary, _ = await yandex_generate_text(summary_prompt)
                context.chat_data["vk_summary"] = summary or ""
            else:
                context.chat_data["vk_summary"] = ""
        await context.bot.send_message(chat_id=chat_id, text="✅ Данные VK проанализированы и сохранены.", reply_markup=main_menu_keyboard(context.chat_data))
    else:
        context.chat_data["vk_summary"] = ""
        await context.bot.send_message(chat_id=chat_id, text="VK не будет использоваться.", reply_markup=main_menu_keyboard(context.chat_data))

    context.user_data.pop("vk_pending_domain", None)

# ----------------- Меню (inline callback) с инструкциями -----------------
async def menu_callback(update: Update, context: CallbackContext):
    q = update.callback_query
    await q.answer()
    try:
        await q.message.delete()
    except Exception as e:
        logger.debug("Не удалось удалить сообщение: %s", e)
    data = q.data
    chat_id = q.message.chat_id

    INSTRUCTIONS = {
        "menu_generate": "Вы можете попросить ИИ-ассистента придумать текст для вашего поста по вашей теме.",
        "menu_editing": "Напишите ваш уже готовый текст поста, бот исправит в нём все ошибки и нормализует его по всем правилам русского языка.",
        "menu_plan": "Может создать план по продвижению, чтобы у вас был чёткий план, как достичь цели.",
        "menu_data": "Заполните информацию о вашем НКО для того, чтобы добиться конкретики, а также для более точного создания постов."
    }

    if data == "main_menu":
        await show_main_menu(update, context)
        return
    elif data == "menu_generate":
        await context.bot.send_message(chat_id=chat_id, text=INSTRUCTIONS["menu_generate"])
        await context.bot.send_message(
            chat_id=chat_id,
            text="Выберите, что сгенерировать:",
            reply_markup=generate_sub_keyboard()
        )
        return
    elif data == "menu_editing":
        await context.bot.send_message(chat_id=chat_id, text=INSTRUCTIONS["menu_editing"])
        await context.bot.send_message(
            chat_id=chat_id,
            text="Что хотите улучшить?",
            reply_markup=editing_sub_keyboard()
        )
        return
    elif data == "menu_plan":
        await context.bot.send_message(chat_id=chat_id, text=INSTRUCTIONS["menu_plan"])
        await context.bot.send_message(
            chat_id=chat_id,
            text="Составим контент-план на несколько дней.",
            reply_markup=plan_sub_keyboard()
        )
        return
    elif data == "menu_data":
        await context.bot.send_message(chat_id=chat_id, text=INSTRUCTIONS["menu_data"])
        await context.bot.send_message(
            chat_id=chat_id,
            text="Управление информацией о вашей НКО:",
            reply_markup=data_sub_keyboard()
        )
        return
    elif data == "plan_7":
        context.user_data["expect"] = "plan"
        context.user_data["plan_days"] = 7
        await handle_plan_generation(update, context, 7)
        return
    elif data == "plan_30":
        context.user_data["expect"] = "plan"
        context.user_data["plan_days"] = 30
        await handle_plan_generation(update, context, 30)
        return
    elif data == "menu_plan_custom":
        await context.bot.send_message(
            chat_id=chat_id,
            text="На сколько дней составить контент-план? (введите число):",
            reply_markup=back_button("menu_plan")
        )
        context.user_data["expect"] = "plan"
        return
    if data == "menu_text":
        await context.bot.send_message(
            chat_id=chat_id,
            text="Введите идею поста (коротко):",
            reply_markup=back_button("menu_generate")
        )
        context.user_data["expect"] = "text"
    elif data == "menu_image":
        await context.bot.send_message(
            chat_id=chat_id,
            text="Опишите изображение (стили, объекты, композиция):",
            reply_markup=back_button("menu_generate")
        )
        context.user_data["expect"] = "image"
    elif data == "menu_edit":
        await context.bot.send_message(
            chat_id=chat_id,
            text="Отправьте текст, который нужно исправить:",
            reply_markup=back_button("menu_editing")
        )
        context.user_data["expect"] = "edit"
    elif data == "menu_info":
        await context.bot.send_message(
            chat_id=chat_id,
            text="Давайте введём данные о вашей НКО. Вы можете в любой момент отправить «Пропустить», чтобы не указывать поле.\n"
                 "1/6. Введите название организации (или «Пропустить»):",
            reply_markup=back_button("menu_data"),
        )
        context.user_data["expect"] = "org_name"
    elif data == "menu_show_info":
        cd = context.chat_data
        info_lines = []
        info_lines.append(f"Название: {cd.get('name','(не указано)')}")
        info_lines.append(f"Описание: {cd.get('description','(не указано)')}")
        info_lines.append(f"Формы деятельности: {cd.get('org_activity','(не указано)')}")
        info_lines.append(f"Целевая аудитория: {cd.get('org_audience','(не указано)')}")
        info_lines.append(f"Локация: {cd.get('org_location','(не указано)')}")
        info_lines.append(f"Контакты: {cd.get('org_contact','(не указано)')}")
        await context.bot.send_message(
            chat_id=chat_id,
            text="Текущие данные НКО:\n" + "\n".join(info_lines),
            reply_markup=main_menu_keyboard(cd)
        )
    elif data == "menu_clear_info":
        for k in ("name", "description", "org_activity", "org_audience", "org_location", "org_contact", "vk_posts", "vk_summary"):
            if k in context.chat_data:
                del context.chat_data[k]
        await context.bot.send_message(
            chat_id=chat_id,
            text="Данные НКО очищены.",
            reply_markup=main_menu_keyboard(context.chat_data)
        )
    elif data == "menu_help":
        await help_command(update, context)
    return

# ----------------- Генерация плана -----------------
async def handle_plan_generation(update: Update, context: CallbackContext, days: int):
    context.user_data.pop("expect", None)
    name = context.chat_data.get("name", "НКО")
    activity = context.chat_data.get("org_activity", "")
    prompt = (
        f"Составь детальный контент-план для НКО '{name}' на {days} дней.\n"
        f"Формы деятельности: {activity}\n"
        "На каждый день: тема, формат поста, краткое описание (1-2 предложения). Верни удобочитаемый список."
    )
    chat_id = update.effective_chat.id
    await context.bot.send_message(chat_id=chat_id, text="Генерирую контент-план...")
    text_out, err = await yandex_generate_text(prompt)
    if err:
        await context.bot.send_message(chat_id=chat_id, text=f"Ошибка: {err}", reply_markup=main_menu_keyboard(context.chat_data))
    else:
        await context.bot.send_message(chat_id=chat_id, text=text_out, reply_markup=plan_result_keyboard())

# ----------------- Показ ПОЛНОГО поста -----------------
async def show_last_post_handler(update: Update, context: CallbackContext):
    chat_data = context.chat_data
    text = chat_data.get("last_post_text")
    image = chat_data.get("last_post_image")
    if not text and not image:
        await update.message.reply_text("Пока что нет сгенерированного поста. Сначала создайте текст и/или изображение.")
        return
    if image is not None:
        bio = io.BytesIO(image)
        bio.name = "post_image.png"
        bio.seek(0)
        if text:
            caption = (text[:1024] + "...") if len(text) > 1024 else text
            await update.message.reply_photo(photo=bio, caption=caption)
        else:
            await update.message.reply_photo(photo=bio)
    elif text:
        await update.message.reply_text(text)

# ----------------- Повторная генерация -----------------
async def regenerate_handler(update: Update, context: CallbackContext):
    logger.info("🔄 regenerate_handler вызван!")
    query = update.callback_query
    await query.answer()
    try:
        await query.message.delete()
    except:
        pass
    chat_id = update.effective_chat.id
    gen_type = context.chat_data.get("last_generation_type")
    prompt = context.chat_data.get("last_prompt")
    if not gen_type or not prompt:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Нет данных для повторной генерации. Сначала создайте пост.",
            reply_markup=main_menu_keyboard(context.chat_data)
        )
        return
    await context.bot.send_message(chat_id=chat_id, text="🔄 Генерирую новый вариант...")
    try:
        if gen_type == "text":
            name = context.chat_data.get("name", "")
            desc = context.chat_data.get("description", "")
            activity = context.chat_data.get("org_activity", "")
            audience = context.chat_data.get("org_audience", "")
            location = context.chat_data.get("org_location", "")
            vk_summary = context.chat_data.get("vk_summary", "")
            parts = [f"НКО: {name}" if name else "НКО: (не указано)"]
            if desc: parts.append(f"Описание: {desc}")
            if activity: parts.append(f"Формы деятельности: {activity}")
            if audience: parts.append(f"Целевая аудитория: {audience}")
            if location: parts.append(f"Локация: {location}")
            if vk_summary: parts.append(f"Стиль из VK: {vk_summary}")
            full_prompt = "\n".join(parts) + f"\nИдея поста: {prompt}\n"
            full_prompt += "Напишите готовый пост для соцсетей: с призывом к действию и хэштегами."
            text_out, err = await yandex_generate_text(full_prompt)
            if err:
                await context.bot.send_message(chat_id=chat_id, text=f"❌ Ошибка: {err}")
            else:
                context.chat_data["last_post_text"] = text_out
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=text_out,
                    reply_markup=regenerate_keyboard()
                )
        elif gen_type == "image":
            meta = []
            if context.chat_data.get("name"):
                meta.append(f"Организация: {context.chat_data.get('name')}")
            if context.chat_data.get("org_activity"):
                meta.append(f"Формы деятельности: {context.chat_data.get('org_activity')}")
            full_prompt = f"{prompt}\n" + " | ".join(meta) if meta else prompt
            img, err = await yandex_generate_image(full_prompt, max_poll_seconds=90)
            if err:
                await context.bot.send_message(chat_id=chat_id, text=f"❌ Ошибка: {err}")
            else:
                context.chat_data["last_post_image"] = img
                bio = io.BytesIO(img)
                bio.name = "image.png"
                bio.seek(0)
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=bio,
                    reply_markup=regenerate_keyboard()
                )
    except Exception as e:
        logger.exception("Ошибка в regenerate_handler")
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Ошибка: {e}")

# ----------------- Общий обработчик текста -----------------
async def generic_handler(update: Update, context: CallbackContext):
    expect = context.user_data.get("expect")
    chat_id = update.effective_chat.id
    await send_chat_action(context, chat_id, ChatAction.TYPING)
    if not (update.message and update.message.text):
        await update.message.reply_text("Пожалуйста, отправьте текст.", reply_markup=main_menu_keyboard(context.chat_data))
        return
    text_msg = update.message.text.strip()
    if text_msg == "✨ Создать пост":
        await update.message.reply_text(BUTTON_DESCRIPTIONS["✨ Создать пост"])
        await update.message.reply_text(
            "Выберите, что сгенерировать:",
            reply_markup=generate_post_options_keyboard()
        )
        return
    elif text_msg == "✍️ Исправить текст":
        await update.message.reply_text(BUTTON_DESCRIPTIONS["✍️ Исправить текст"])
        context.user_data["expect"] = "edit"
        await update.message.reply_text(
            "Отправьте текст, который нужно исправить:",
            reply_markup=back_button("menu_editing")
        )
        return
    elif text_msg == "📄 Пост":
        await update.message.reply_text(BUTTON_DESCRIPTIONS["📄 Пост"])
        await show_last_post_handler(update, context)
        return

    # --- Пошаговый ввод НКО ---
    if expect and isinstance(expect, str) and expect.startswith("org_"):
        if expect == "org_name":
            val = "" if text_msg.lower() == "пропустить" else text_msg
            context.chat_data["name"] = val
            context.user_data["expect"] = "org_desc"
            await update.message.reply_text("2/6. Краткое описание НКО или «Пропустить»:", reply_markup=back_button("menu_data"))
            return
        if expect == "org_desc":
            val = "" if text_msg.lower() == "пропустить" else text_msg
            context.chat_data["description"] = val
            context.user_data["expect"] = "org_activity"
            await update.message.reply_text("3/6. Формы деятельности или «Пропустить»:", reply_markup=back_button("menu_data"))
            return
        if expect == "org_activity":
            val = "" if text_msg.lower() == "пропустить" else text_msg
            context.chat_data["org_activity"] = val
            context.user_data["expect"] = "org_audience"
            await update.message.reply_text("4/6. Целевая аудитория или «Пропустить»:", reply_markup=back_button("menu_data"))
            return
        if expect == "org_audience":
            val = "" if text_msg.lower() == "пропустить" else text_msg
            context.chat_data["org_audience"] = val
            context.user_data["expect"] = "org_location"
            await update.message.reply_text("5/6. Локация / регион работы или «Пропустить»:", reply_markup=back_button("menu_data"))
            return
        if expect == "org_location":
            val = "" if text_msg.lower() == "пропустить" else text_msg
            context.chat_data["org_location"] = val
            context.user_data["expect"] = "org_contact"
            await update.message.reply_text("6/6. Контакты (сайт, соцсети и т.д.) или «Пропустить»:", reply_markup=back_button("menu_data"))
            return
        if expect == "org_contact":
            val = "" if text_msg.lower() == "пропустить" else text_msg
            context.chat_data["org_contact"] = val
            urls = re.findall(r"(https?://[^\s]+|vk\.com/[^\s]+)", val)
            vk_domain = None
            for u in urls:
                if not u.startswith("http"):
                    u = "https://" + u
                if "vk.com" in u:
                    d = extract_vk_domain(u)
                    if d:
                        vk_domain = d
                        break
            if vk_domain:
                context.user_data["vk_pending_domain"] = vk_domain
                kb = InlineKeyboardMarkup([
                    [InlineKeyboardButton("Да, можно (публичные посты)", callback_data="vk_confirm_yes"),
                     InlineKeyboardButton("Нет", callback_data="vk_confirm_no")]
                ])
                await update.message.reply_text(
                    f"В контактах найдена ссылка на VK: {vk_domain}.\n"
                    "Разрешаете боту получить последние публичные посты для анализа стиля?",
                    reply_markup=kb,
                )
                context.user_data.pop("expect", None)
                return
            else:
                cd = context.chat_data
                lines = [f"{k}: {cd.get(k, '(не указано)')}" for k in ("name", "description", "org_activity", "org_audience", "org_location", "org_contact")]
                await update.message.reply_text("Спасибо! Данные сохранены:\n" + "\n".join(lines), reply_markup=main_menu_keyboard(cd))
                context.user_data.pop("expect", None)
                return

    # --- Генерация текста ---
    if expect == "text":
        context.user_data.pop("expect", None)
        idea = text_msg
        if is_topic_inappropriate(idea):
            await update.message.reply_text("Простите, но посты на такие темы я не генерирую.", reply_markup=main_menu_keyboard(context.chat_data))
            return
        context.chat_data["last_prompt"] = idea
        context.chat_data["last_generation_type"] = "text"
        name = context.chat_data.get("name", "")
        desc = context.chat_data.get("description", "")
        activity = context.chat_data.get("org_activity", "")
        audience = context.chat_data.get("org_audience", "")
        location = context.chat_data.get("org_location", "")
        vk_summary = context.chat_data.get("vk_summary", "")
        parts = [f"НКО: {name}" if name else "НКО: (не указано)"]
        if desc: parts.append(f"Описание: {desc}")
        if activity: parts.append(f"Формы деятельности: {activity}")
        if audience: parts.append(f"Целевая аудитория: {audience}")
        if location: parts.append(f"Локация: {location}")
        if vk_summary: parts.append(f"Стиль из VK: {vk_summary}")
        prompt = "\n".join(parts) + f"\nИдея поста: {idea}\n"
        prompt += "Напишите готовый пост для соцсетей: с призывом к действию и хэштегами."
        await update.message.reply_text("Формируем текст...")
        text_out, err = await yandex_generate_text(prompt)
        if err:
            await update.message.reply_text(f"Ошибка: {err}", reply_markup=main_menu_keyboard(context.chat_data))
        else:
            context.chat_data["last_post_text"] = text_out
            await update.message.reply_text(text_out, reply_markup=regenerate_keyboard())
        return

    # --- Генерация изображения ---
    if expect == "image":
        context.user_data.pop("expect", None)
        desc = text_msg
        context.chat_data["last_prompt"] = desc
        context.chat_data["last_generation_type"] = "image"
        meta = []
        if context.chat_data.get("name"):
            meta.append(f"Организация: {context.chat_data.get('name')}")
        if context.chat_data.get("org_activity"):
            meta.append(f"Формы деятельности: {context.chat_data.get('org_activity')}")
        full_prompt = f"{desc}\n" + " | ".join(meta) if meta else desc
        await update.message.reply_text("Генерируем изображение... (до 1 мин)")
        await send_chat_action(context, chat_id, ChatAction.UPLOAD_PHOTO)
        img, err = await yandex_generate_image(full_prompt, max_poll_seconds=90)
        if err:
            await update.message.reply_text(f"Ошибка: {err}", reply_markup=main_menu_keyboard(context.chat_data))
        else:
            context.chat_data["last_post_image"] = img
            bio = io.BytesIO(img)
            bio.name = "image.png"
            bio.seek(0)
            await update.message.reply_photo(photo=bio, reply_markup=regenerate_keyboard())
        return

    # --- Исправление текста ---
    if expect == "edit":
        context.user_data.pop("expect", None)
        original = text_msg
        await update.message.reply_text("Исправляю текст...")
        corrected, err = await check_text_with_yandex(original)
        if err:
            await update.message.reply_text(f"Ошибка: {err}", reply_markup=main_menu_keyboard(context.chat_data))
        else:
            await update.message.reply_text("Исправленный текст:\n" + corrected, reply_markup=main_menu_keyboard(context.chat_data))
        return

    # --- Контент-план ---
    if expect == "plan":
        context.user_data.pop("expect", None)
        try:
            days = int(text_msg)
            if days <= 0:
                raise ValueError
        except Exception:
            await update.message.reply_text("Введите корректное положительное число дней.", reply_markup=back_button("menu_plan"))
            return
        await handle_plan_generation(update, context, days)
        return

    # --- По умолчанию ---
    await update.message.reply_text(
        "Используйте главное меню или команды:\n/text /image /edit /plan /orginfo /help",
        reply_markup=main_menu_keyboard(context.chat_data)
    )

# ----------------- Исправление текста -----------------
async def check_text_with_yandex(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not YANDEX_API_KEY:
        return None, "YANDEX_API_KEY не установлен!"
    prompt = (
        "Исправь орфографию, пунктуацию и стилистические ошибки в тексте, "
        "сохранив смысл. Верни только исправленную версию текста.\n"
        f"Текст: {text}"
    )
    return await yandex_generate_text(prompt)

# ----------------- MAIN -----------------
def main():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN не установлен!")
        return
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("menu", menu_command))
    app.add_handler(CommandHandler("text", text_command))
    app.add_handler(CommandHandler("image", image_command))
    app.add_handler(CommandHandler("edit", edit_command))
    app.add_handler(CommandHandler("plan", plan_command))
    app.add_handler(CommandHandler("orginfo", orginfo_command))
    # Обработчики
    app.add_handler(CallbackQueryHandler(regenerate_handler, pattern="^regenerate$"))
    app.add_handler(CallbackQueryHandler(fake_reminder_handler, pattern="^fake_reminder$"))
    app.add_handler(CallbackQueryHandler(vk_confirm_handler, pattern="^vk_confirm_(yes|no)$"))
    app.add_handler(CallbackQueryHandler(menu_callback))
    # Обычные сообщения
    app.add_handler(MessageHandler(filters.Regex("^Показать готовый пост$"), show_last_post_handler))
    app.add_handler(MessageHandler(filters.Regex("^✍️ Исправить текст$"), generic_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generic_handler))
    logger.info("✅ Бот запущен!")
    app.run_polling(allowed_updates=None)

if __name__ == "__main__":
    main()