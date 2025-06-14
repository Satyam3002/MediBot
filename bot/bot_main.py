from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import os
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.diagnosis_text import diagnosis_system

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🩺 Hello! I'm MediBot.\n\n"
        "I can help you get a preliminary diagnosis based on your symptoms.\n"
        "Please describe your symptoms in detail, and I'll try to help you understand what might be wrong.\n\n"
        "⚠️ Note: This is not a replacement for professional medical advice. Always consult a doctor for proper diagnosis and treatment."
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    
    # Get diagnosis from the system
    diagnosis = diagnosis_system.get_diagnosis(user_input)
    
    # Send response to user
    await update.message.reply_text(
        f"📋 Based on your symptoms: '{user_input}'\n\n"
        f"{diagnosis}\n\n"
        "⚠️ Remember: This is a preliminary assessment. Please consult a healthcare professional for proper medical advice."
    )

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()