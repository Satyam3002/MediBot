from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler
import os
from dotenv import load_dotenv
import sys
import httpx
import io
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.diagnosis_text import diagnosis_system

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = "http://127.0.0.1:8000"  # FastAPI backend URL

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🩺 Hello! I'm MediBot.\n\n"
        "I can help you in three ways:\n"
        "1. Text-based diagnosis: Describe your symptoms in detail\n"
        "2. X-ray analysis: Send me an X-ray image\n"
        "3. Skin condition analysis: Send me a photo of your skin condition\n\n"
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

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Get the photo file
        photo = await update.message.photo[-1].get_file()
        photo_bytes = await photo.download_as_bytearray()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(photo_bytes))
        
        # Save to bytes for API request
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Store the image in user context for later use
        context.user_data['last_image'] = img_byte_arr
        
        # Create inline keyboard with buttons
        keyboard = [
            [
                InlineKeyboardButton("X-ray Analysis", callback_data='xray'),
                InlineKeyboardButton("Skin Analysis", callback_data='skin')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Ask user what type of analysis they want
        await update.message.reply_text(
            "What type of analysis would you like?",
            reply_markup=reply_markup
        )
        
    except Exception as e:
        await update.message.reply_text(f"Sorry, I couldn't process your image. Error: {str(e)}")

async def handle_analysis_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if 'last_image' not in context.user_data:
        await query.edit_message_text("Please send an image first!")
        return
    
    analysis_type = query.data  # This will be either 'xray' or 'skin'
    
    try:
        # Prepare the API request
        files = {'file': ('image.jpg', context.user_data['last_image'], 'image/jpeg')}
        endpoint = f"{API_URL}/diagnose-{analysis_type}"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, files=files)
            
        if response.status_code == 200:
            result = response.json()
            condition = result['prediction']
            
            # Detailed medical recommendations for X-ray conditions
            xray_recommendations = {
                "Edema": {
                    "severity": "Moderate",
                    "description": "Fluid accumulation in the lungs or tissues",
                    "immediate_actions": [
                        "Seek medical attention if experiencing shortness of breath",
                        "Keep track of symptoms and their progression",
                        "Monitor oxygen levels if you have a pulse oximeter"
                    ],
                    "treatment": [
                        "Diuretics may be prescribed by your doctor",
                        "Limit salt intake",
                        "Elevate legs if lower body edema",
                        "Wear compression stockings if recommended"
                    ],
                    "follow_up": "Schedule a follow-up with your doctor within 1-2 weeks"
                },
                "Cardiomegaly": {
                    "severity": "High",
                    "description": "Enlarged heart",
                    "immediate_actions": [
                        "Seek immediate medical attention if experiencing chest pain",
                        "Monitor blood pressure regularly",
                        "Keep track of any new symptoms"
                    ],
                    "treatment": [
                        "Medications to manage heart function",
                        "Lifestyle modifications (diet, exercise)",
                        "Regular cardiac monitoring"
                    ],
                    "follow_up": "Schedule cardiology appointment within 1 week"
                },
                "Pneumonia": {
                    "severity": "High",
                    "description": "Lung infection",
                    "immediate_actions": [
                        "Seek immediate medical attention",
                        "Monitor temperature",
                        "Rest and stay hydrated"
                    ],
                    "treatment": [
                        "Antibiotics as prescribed",
                        "Rest and adequate hydration",
                        "Deep breathing exercises"
                    ],
                    "follow_up": "Follow up with doctor in 2-3 days"
                },
                "Consolidation": {
                    "severity": "Moderate to High",
                    "description": "Solidification of lung tissue",
                    "immediate_actions": [
                        "Seek medical attention",
                        "Monitor breathing patterns",
                        "Keep track of symptoms"
                    ],
                    "treatment": [
                        "Antibiotics if bacterial",
                        "Rest and proper nutrition",
                        "Avoid smoking and secondhand smoke"
                    ],
                    "follow_up": "Schedule follow-up within 1 week"
                },
                "No Finding": {
                    "severity": "Low",
                    "description": "No significant abnormalities detected",
                    "immediate_actions": [
                        "Continue monitoring any symptoms",
                        "Maintain regular health check-ups"
                    ],
                    "treatment": [
                        "Continue any existing prescribed treatments",
                        "Maintain healthy lifestyle"
                    ],
                    "follow_up": "Regular check-up as per doctor's schedule"
                }
            }
            
            # Get recommendations for the condition
            if condition in xray_recommendations:
                rec = xray_recommendations[condition]
                response_text = (
                    f"🔍 Analysis Results:\n\n"
                    f"Condition: {condition}\n"
                    f"Severity: {rec['severity']}\n"
                    f"Description: {rec['description']}\n\n"
                    f"Immediate Actions:\n" + "\n".join(f"• {action}" for action in rec['immediate_actions']) + "\n\n"
                    f"Treatment Recommendations:\n" + "\n".join(f"• {treatment}" for treatment in rec['treatment']) + "\n\n"
                    f"Follow-up: {rec['follow_up']}\n\n"
                    "⚠️ Remember: This is a preliminary assessment. Please consult a healthcare professional for proper medical advice."
                )
            else:
                response_text = (
                    f"🔍 Analysis Results:\n\n"
                    f"Predicted condition: {condition}\n\n"
                    "⚠️ Remember: This is a preliminary assessment. Please consult a healthcare professional for proper medical advice."
                )
            
            await query.edit_message_text(response_text)
        else:
            await query.edit_message_text("Sorry, I couldn't analyze the image. Please try again.")
            
    except Exception as e:
        await query.edit_message_text(f"Sorry, an error occurred: {str(e)}")
    finally:
        # Clean up
        if 'last_image' in context.user_data:
            del context.user_data['last_image']

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(CallbackQueryHandler(handle_analysis_callback))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()