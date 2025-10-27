import os
import json
import re
import requests
import base64
from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update, error
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# === NLTK setup ===
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN не установлен в .env")

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/chat")  # ← ВАЖНО: /api/chat, не /generate
MODEL_NAME = os.getenv("MODEL_NAME", "gemma3:4b")

FAQ_PATH = "faq.json"
knowledge_base = []
faq_texts = []

if os.path.exists(FAQ_PATH):
    try:
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            knowledge_base = [
                {
                    "context": item.get("context", ""),
                    "keywords": [kw.lower().strip() for kw in item.get("keywords", [])]
                }
                for item in raw_data
                if item.get("context")
            ]
        faq_texts = [
            " ".join(entry["keywords"]) + " " + entry["context"]
            for entry in knowledge_base
        ]
        print(f"✅ Загружено {len(knowledge_base)} записей для RAG")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки базы знаний: {e}")

stemmer = PorterStemmer()
try:
    russian_stopwords = set(stopwords.words('russian'))
except:
    russian_stopwords = set()

def preprocess_text(text: str) -> str:
    text = re.sub(r"[^а-яёa-z\s]", " ", text.lower())
    words = text.split()
    if russian_stopwords:
        words = [w for w in words if w not in russian_stopwords]
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), lowercase=False, max_features=1000)
tfidf_matrix = None
if faq_texts:
    processed_faq = [preprocess_text(txt) for txt in faq_texts]
    tfidf_matrix = tfidf.fit_transform(processed_faq)

def retrieve_context(user_question: str, similarity_threshold: float = 0.1) -> str:
    if not knowledge_base:
        return ""
    norm_question = preprocess_text(user_question)
    for entry in knowledge_base:
        for kw in entry["keywords"]:
            if preprocess_text(kw) in norm_question:
                return entry["context"]
    if tfidf_matrix is not None:
        query_vec = tfidf.transform([norm_question])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        best_idx = similarities.argmax()
        if similarities[best_idx] >= similarity_threshold:
            return knowledge_base[best_idx]["context"]
    return ""

# === НОВАЯ ФУНКЦИЯ: запрос к Ollama через /api/chat с поддержкой изображений ===
def ask_model_ollama(messages: list) -> str:
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.9}
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "Не удалось получить ответ.")
    except Exception as e:
        return f"Ошибка при обращении к модели: {str(e)}"

# === Преобразование изображения в base64 ===
def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

TEACHER_ROLE = (
    "Ты личный ассистент репетитора по программированию Алексея. (@AVick23) "
    "Отвечай только на вопросы о его курсах, уроках, обучении, программировании, IT, "
    "а также на 'Кто такой Алексей?' и 'Кто ты?'. "
    "Ответ должен быть кратким — 1–2 предложения, без повторения вопроса. "
    "Говори спокойно, вежливо и по-человечески, но без эмодзи, сленга и лишних слов. "
    "Если тема не относится к указанным — вежливо откажись отвечать."
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_msg = (
        "👨‍🏫 Привет! Я — ваш гид по обучению.\n"
        "Могу отвечать на вопросы по курсам, а также понимать текст и изображения.\n\n"
        "Введите 'очистить', чтобы начать новый диалог."
    )
    await update.message.reply_text(welcome_msg)

# === Обработка текста ===
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text.strip()
    if user_message.lower() in ("очистить", "clear", "сбросить"):
        context.user_data['chat_history'] = []
        await update.message.reply_text("✅ Диалог очищен.")
        return

    context.user_data.setdefault('chat_history', [])
    context.user_data['chat_history'].append({"role": "user", "content": user_message})

    retrieved_context = retrieve_context(user_message)
    if retrieved_context:
        system_msg = {"role": "system", "content": f"Контекст из базы знаний: {retrieved_context}"}
        messages = [system_msg] + context.user_data['chat_history'][-6:]
    else:
        messages = [{"role": "system", "content": TEACHER_ROLE}] + context.user_data['chat_history'][-6:]

    thinking = await update.message.reply_text("🤔 Думаю...")
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, lambda: ask_model_ollama(messages))
    await thinking.delete()
    
    context.user_data['chat_history'].append({"role": "assistant", "content": response})
    await update.message.reply_text(response)

# === Обработка изображений ===
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    file_path = f"temp_{photo.file_id}.jpg"
    await file.download_to_drive(file_path)

    try:
        # Конвертируем в base64
        img_b64 = image_to_base64(file_path)

        # Добавляем сообщение с изображением
        user_msg = {"role": "user", "content": "Проанализируй это изображение.", "images": [img_b64]}
        context.user_data.setdefault('chat_history', []).append(user_msg)

        # Формируем контекст для модели
        messages = [{"role": "system", "content": TEACHER_ROLE}] + context.user_data['chat_history'][-6:]

        thinking = await update.message.reply_text("🖼️ Анализирую изображение...")
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: ask_model_ollama(messages))
        await thinking.delete()

        context.user_data['chat_history'].append({"role": "assistant", "content": response})
        await update.message.reply_text(response)

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def main():
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))  # ← обработка фото
    print("🤖 Бот запущен с поддержкой текста и изображений через Ollama.")
    application.run_polling()

if __name__ == "__main__":
    main()