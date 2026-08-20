import os
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask
from threading import Thread
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, MessageHandler, Filters
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# ======================= МОДЕЛ (вграден) =======================
class SalesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'den_v_nedela', 'mesec', 'dali_praznik',
            'prodazba_prethoden_den', 'prosecna_prodazba_posledni_7'
        ]
    
    def _create_features(self, df):
        df = df.copy()
        df['den_v_nedela'] = pd.to_datetime(df['datum']).dt.dayofweek
        df['mesec'] = pd.to_datetime(df['datum']).dt.month
        df['dali_praznik'] = df['den_v_nedela'].isin([5, 6]).astype(int)
        df['prodazba_prethoden_den'] = df.groupby('proizvod')['prodadeno_kolicestvo'].shift(1)
        df['prosecna_prodazba_posledni_7'] = df.groupby('proizvod')['prodadeno_kolicestvo'].transform(lambda x: x.rolling(7, min_periods=1).mean())
        df = df.fillna(0)
        return df
    
    def train(self, data_path):
        df = pd.read_csv(data_path)
        df['datum'] = pd.to_datetime(df['datum'])
        df = df.sort_values(['proizvod', 'datum'])
        df = self._create_features(df)
        X = df[self.feature_columns].values
        y = df['prodadeno_kolicestvo'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"✅ Моделот трениран! Грешка: {mae:.2f} единици")
        return mae
    
    def predict(self, proizvod, den_v_nedela, mesec, dali_praznik,
                prodazba_prethoden_den, prosecna_prodazba_posledni_7, *args):
        if self.model is None:
            raise Exception("Моделот не е трениран!")
        X = np.array([[den_v_nedela, mesec, dali_praznik,
                       prodazba_prethoden_den, prosecna_prodazba_posledni_7]])
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)[0]
        return max(0, round(pred))
    
    def predict_visok(self, na_zaliha, predvidena_prodazba, rok_denovi=30):
        if na_zaliha <= predvidena_prodazba:
            return {'ima_visok': False, 'visok_kolicina': 0, 'preporaka_akcija': "✅ Нема вишок.", 'preporaka_popust': 0}
        visok = na_zaliha - predvidena_prodazba
        if rok_denovi <= 2:
            popust = 50
            akcija = f"🚨 ИТНО! Истекува за {rok_denovi} дена. Попуст {popust}%!"
        elif rok_denovi <= 5:
            popust = 30
            akcija = f"⚠️ Истекува за {rok_denovi} дена. Попуст {popust}%. Вишок: {visok}."
        else:
            popust = 15
            akcija = f"ℹ️ Рок {rok_denovi} дена. Попуст {popust}% за вишок од {visok}."
        return {'ima_visok': True, 'visok_kolicina': visok, 'preporaka_akcija': akcija, 'preporaka_popust': popust}
    
    def save(self, path):
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'feature_columns': self.feature_columns}, path)
        print(f"✅ Моделот зачуван во {path}")
    
    def load(self, path):
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']
            print(f"✅ Моделот вчитан од {path}")
            return True
        return False

# ======================= ТОКЕН =======================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не е поставен во Environment Variables.")

DATA_PATH = "data/prodazbi.csv"
MODEL_PATH = "model_prodazba.joblib"

# ======================= ИНИЦИЈАЛИЗАЦИЈА =======================
predictor = SalesPredictor()
if not predictor.load(MODEL_PATH):
    if os.path.exists(DATA_PATH):
        print("🔄 Тренирам нов модел со свежите податоци...")
        predictor.train(DATA_PATH)
        predictor.save(MODEL_PATH)
        print("✅ Моделот е претрениран!")
    else:
        print("❌ Нема податоци! Стави CSV во data/prodazbi.csv")

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