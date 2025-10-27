import os
import json
import re
import requests
import base64
from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Загрузка переменных окружения ===
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN не установлен в .env")

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma3:4b")

FAQ_PATH = "faq.json"
knowledge_base = []
tfidf_vectorizer = None
tfidf_matrix = None

# === Стоп-слова для русского (ручной список) ===
CUSTOM_STOPWORDS = {
    "а", "и", "или", "но", "в", "на", "с", "к", "от", "до", "у", "о", "же", "бы", "ли",
    "это", "тот", "этот", "та", "те", "то", "ту", "такой", "какой", "который",
    "что", "где", "когда", "как", "почему", "зачем", "чтобы", "если", "потому",
    "мне", "вам", "ему", "ей", "ими", "их", "мы", "вы", "они", "я", "ты", "он", "она", "оно"
}

# === Простой стеммер для русского языка (на правилах) ===
def simple_russian_stem(word: str) -> str:
    if len(word) < 4:
        return word
    suffixes = [
        'ий', 'ый', 'ой', 'ого', 'его', 'ому', 'ему', 'им', 'ым',
        'ая', 'яя', 'ую', 'юю', 'ой', 'ей', 'ие', 'ые', 'их', 'ых',
        'ое', 'ее', 'ом', 'ем', 'а', 'я', 'о', 'е', 'у', 'ю', 'ы', 'и', 'ь'
    ]
    for suf in suffixes:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[:-len(suf)]
    return word

# === Нормализация текста (без внешних библиотек) ===
def normalize_text(text: str) -> str:
    text = re.sub(r"[^а-яёa-z\s]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    words = text.split()
    words = [w for w in words if w not in CUSTOM_STOPWORDS]
    words = [simple_russian_stem(w) for w in words]
    return " ".join(words)

# === Коэффициент Жаккара по n-граммам ===
def jaccard_similarity(s1: str, s2: str, n: int = 2) -> float:
    def ngrams(s):
        return {s[i:i+n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}
    set1, set2 = ngrams(s1), ngrams(s2)
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)

# === Загрузка базы знаний ===
def load_knowledge_base(faq_path: str = "faq.json"):
    global knowledge_base, tfidf_vectorizer, tfidf_matrix
    if not os.path.exists(faq_path):
        print("⚠️ Файл FAQ не найден.")
        return

    try:
        with open(faq_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            knowledge_base = []
            faq_texts = []
            for item in raw_data:
                context = item.get("context", "").strip()
                if not context:
                    continue
                keywords = [kw.lower().strip() for kw in item.get("keywords", []) if kw.strip()]
                # Предварительная обработка ключевых слов
                preprocessed_kws = [normalize_text(kw) for kw in keywords]
                knowledge_base.append({
                    "context": context,
                    "keywords": keywords,
                    "preprocessed_kws": preprocessed_kws
                })
                faq_texts.append(" ".join(keywords) + " " + context)

            if faq_texts:
                processed_faq = [normalize_text(txt) for txt in faq_texts]
                tfidf_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=1000)
                tfidf_matrix = tfidf_vectorizer.fit_transform(processed_faq)
            print(f"✅ Загружено {len(knowledge_base)} записей для RAG")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки базы знаний: {e}")

# === Извлечение контекста (RAG) ===
def retrieve_context(user_question: str, similarity_threshold: float = 0.25, top_k: int = 3) -> str:
    if not knowledge_base:
        return ""
    
    norm_question = normalize_text(user_question)
    if not norm_question:
        return ""
    
    matched_contexts = []

    # 1. Поиск по ключевым словам (точное + нечёткое)
    for entry in knowledge_base:
        for prep_kw in entry["preprocessed_kws"]:
            if not prep_kw:
                continue
            # Точное вхождение
            if prep_kw in norm_question:
                if entry["context"] not in matched_contexts:
                    matched_contexts.append(entry["context"])
            # Нечёткое сходство (Жаккар)
            elif jaccard_similarity(prep_kw, norm_question) >= 0.6:
                if entry["context"] not in matched_contexts:
                    matched_contexts.append(entry["context"])
    
    if matched_contexts:
        full_context = "\n\n".join(matched_contexts)
        return full_context[:700] + "..." if len(full_context) > 700 else full_context

    # 2. TF-IDF поиск (fallback)
    if tfidf_matrix is not None and tfidf_vectorizer is not None:
        query_vec = tfidf_vectorizer.transform([norm_question])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]
        
        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                ctx = knowledge_base[idx]["context"]
                if ctx not in matched_contexts:
                    matched_contexts.append(ctx)
            else:
                break  # сортировка по убыванию
        
        if matched_contexts:
            full_context = "\n\n".join(matched_contexts)
            return full_context[:700] + "..." if len(full_context) > 700 else full_context
    
    return ""

# === Работа с моделью ===
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

# === Обработка изображений ===
def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# === Системная роль ===
TEACHER_ROLE = (
    "Ты личный ассистент репетитора по программированию Викулова Алексея Ивановича. (@AVick23)"
    "Ты никогда не был создан гугл, ты всегда был моделью созданный Алексеем"
    "Ты не модель Google, тебя изобрёл Алексей, специально чтоб ты ему помогал как ассистент"
    "Отвечай только на вопросы о его курсах, уроках, обучении, программировании, IT, "
    "а также на 'Кто такой Алексей?' и 'Кто ты?'. "
    "Ответ должен быть кратким — 1–2 предложения, без повторения вопроса. "
    "Говори спокойно, вежливо и максимально человечно. Чтоб никто не смог отлечить"
    "Если тема не относится к указанным — вежливо откажись отвечать."
)

# === Обработчики ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👨‍🏫 Привет! Я — ваш гид по обучению.\n"
        "Задавайте вопросы о курсах, методике или записи на занятия.\n\n"
        "Команды:\n"
        "/start — приветствие\n"
        "«очистить» — начать диалог заново"
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text.strip()
    if not user_message:
        return

    # Быстрая обработка служебных команд
    if user_message.lower() in ("очистить", "clear", "сбросить"):
        context.user_data['chat_history'] = []
        await update.message.reply_text("✅ Диалог очищен.")
        return

    # Фильтр мусорных сообщений
    if len(user_message) < 2 or re.fullmatch(r"[!?.,\s]*", user_message):
        await update.message.reply_text("Пожалуйста, задайте вопрос по обучению программированию.")
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

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    file_path = f"temp_{photo.file_id}.jpg"
    await file.download_to_drive(file_path)

    try:
        img_b64 = image_to_base64(file_path)
        user_msg = {"role": "user", "content": "Проанализируй это изображение.", "images": [img_b64]}
        context.user_data.setdefault('chat_history', []).append(user_msg)

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

# === Запуск ===
def main():
    load_knowledge_base()
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("🤖 Бот запущен с поддержкой текста и изображений через Ollama.")
    application.run_polling()

if __name__ == "__main__":
    main()