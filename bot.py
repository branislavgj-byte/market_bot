import os
import pandas as pd
from datetime import datetime
from flask import Flask
from threading import Thread
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, MessageHandler, Filters

# ======================= ТОКЕН =======================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не е поставен во Environment Variables.")

DATA_PATH = "data/prodazbi.csv"
MODEL_PATH = "model_prodazba.joblib"

# ======================= МОДЕЛ =======================
from my_model import SalesPredictor

# Креирај го моделот и тренирај го со новите податоци
predictor = SalesPredictor()

# Провери дали CSV-то е поново од моделот (или моделот не постои)
train_needed = False
if not os.path.exists(MODEL_PATH):
    train_needed = True
else:
    # Ако CSV-то е поново од моделот, тренирај повторно
    if os.path.exists(DATA_PATH):
        csv_time = os.path.getmtime(DATA_PATH)
        model_time = os.path.getmtime(MODEL_PATH)
        if csv_time > model_time:
            train_needed = True

if train_needed:
    print("🔄 Тренирам нов модел со свежите податоци...")
    predictor.train(DATA_PATH)
    predictor.save(MODEL_PATH)
    print("✅ Моделот е претрениран!")
else:
    print("✅ Моделот е веќе ажуриран.")
    predictor.load(MODEL_PATH)

# ======================= ФУНКЦИИ =======================
def read_data():
    df = pd.read_csv(DATA_PATH)
    return df

def get_expiry_report():
    df = read_data()
    denes = datetime.now().date()
    result = []
    for _, row in df.iterrows():
        rok = pd.to_datetime(row['Rok']).date()
        preostanati = (rok - denes).days
        if preostanati <= 30:
            result.append(f"• {row['Produkt']}: {preostanati} дена (до {rok})")
    return "\n".join(result) if result else "📋 Нема производи што истекуваат наскоро."

# ======================= БОТ =======================
def start(update, context):
    update.message.reply_text(
        "🛒 Добредојде во **Market AI**!\n"
        "Користи /menu за опции.",
        parse_mode="Markdown"
    )

def menu(update, context):
    keyboard = [
        [InlineKeyboardButton("📋 Преглед на истекување", callback_data="expiry")],
        [InlineKeyboardButton("❓ Помош", callback_data="help")]
    ]
    update.message.reply_text("Што сакаш?", reply_markup=InlineKeyboardMarkup(keyboard))

def expiry(update, context):
    update.message.reply_text(f"📋 *Преглед на истекување*\n\n{get_expiry_report()}", parse_mode="Markdown")

def handle_callback(update, context):
    query = update.callback_query
    query.answer()
    if query.data == "expiry":
        query.edit_message_text(f"📋 *Преглед на истекување*\n\n{get_expiry_report()}", parse_mode="Markdown")
    elif query.data == "help":
        query.edit_message_text("📌 Користи /menu за опции.")

def handle_message(update, context):
    update.message.reply_text("❌ Непозната команда. Користи /menu.")

# ======================= FLASK =======================
flask_app = Flask('')

@flask_app.route('/')
def health():
    return "🛒 Market AI bot is running!", 200

def run_flask():
    port = int(os.environ.get('PORT', 10000))
    flask_app.run(host='0.0.0.0', port=port)

# ======================= MAIN =======================
def main():
    Thread(target=run_flask, daemon=True).start()
    
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("menu", menu))
    dp.add_handler(CommandHandler("expiry", expiry))
    dp.add_handler(CallbackQueryHandler(handle_callback))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    
    print("🤖 Ботот работи...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    print("=== БОТОТ СТАРТУВА ===")
    main()