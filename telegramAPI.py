# Personal Notebook to deploy PersonaGPT as a telegram chatbot
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import yaml

# Load yaml config
with open("config.yaml", "r") as f:
      config = yaml.safe_load(f)

# Get bot token
TOKEN = str(config["personal_parameters"]["TELEGRAM_API_ID"])

async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'Hello {update.effective_user.first_name}')

app = ApplicationBuilder().token("YOUR TOKEN HERE").build()

app.add_handler(CommandHandler("hello", hello))

app.run_polling()