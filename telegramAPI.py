from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
import yaml
import re

from helper_functions import change_prompts, normalize_text

with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    
TOKEN = config['personal_parameters']['TELEGRAM_API']
q_prompt, finetune_prompt, c_prompt, context_label = change_prompts(language="uk", df=None)

def build_input_output(user_message: str, model_answer: str = None, context=None):
    if not context: 
        context = context_label

    input_text = (
        #f"[{p_prompt}]: {MODEL_PROMPT}" # Don't use finetune_p here if already trained with finetune_prompt
        f"[{q_prompt}]: {user_message}\n"
        f"[{c_prompt}]: {context}\n"
    )
    
    return input_text, context


async def start_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hello! I am a personal GPT chatbot. I am trained to accurately replicate the conversation style of the person I was trained on. Let's chat. ")

async def start_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message_type = update.message.chat.type
    user_message = update.message.text

    print(f"Message type: {message_type}, User message: {user_message}")

    #  ----------------------------------------  AI Logic starts here ----------------------------------------  #
    normalized_message = normalize_text(user_message)

    input_text = build_input_output(user_message=normalized_message)




    

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Sorry, I didn't understand that command. Please type /start to begin the conversation.")
    

if __name__ == "__main__":
    print("Starting the bot...")
    app = ApplicationBuilder().token(token=TOKEN).build()

    # Start the bot
    app.add_handler(CommandHandler("start", start_conversation))

    # Message handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, start_conversation))

    # Error
    app.add_error_handler(error)

    # Run the bot 
    print("Polling...")
    app.run_polling(poll_interval=3)