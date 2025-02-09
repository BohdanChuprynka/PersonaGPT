import sys
import os
# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

from typing import List

from main.helper_functions import change_prompts, normalize_text

with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    

TARGET_MAX_LENGTH = config['training_parameters']['TARGET_MAX_LENGTH']
OUTPUT_DIR = config['training_parameters']['OUTPUT_DIR']
MODEL_NAME = config['training_parameters']['MODEL_NAME']
TOKEN = config['personal_parameters']['TELEGRAM_API']
q_prompt, finetune_prompt, c_prompt, context_label = change_prompts(language="uk", df=None)
model_answer = None
chat_info = [] # [question, model_answer, message_time] 0: user_message, 1: model_answer, 2: message_time

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=TARGET_MAX_LENGTH,
            early_stopping=True,
            do_sample=False,
            penalty_alpha=0.6,
            temperature=0.6, 
            top_k=10, # Default: 50
            top_p=0.8,
            repetition_penalty=0.9,
            low_memory=True # ? Good
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def build_input_output(question: str) -> str:
    # TODO: Maybe implement a way to follow the time difference between messages

    if len(chat_info) > 0: # If there will be at least 1 message in the chat_info list, create a context
        context = ""
        for key, (question_, answer_, *_) in enumerate(chat_info[-20:]):
            context += f" <Q{key + 1}> {question_} <A{key + 1}> {answer_}"
    else: 
        context = f"[{context_label}]: "

    input_text = (
        f"[{q_prompt}]:{question}\n"
        f"[{c_prompt}]: {context}\n"
    )

    return input_text


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message_time = update.message.date
    user_question = update.message.text
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    first_name = update.message.from_user.first_name
    last_name = update.message.from_user.last_name
    chat_id = update.message.chat_id
    message_id = update.message.message_id
    is_bot = update.message.from_user.is_bot
    language_code = update.message.from_user.language_code

    print(f"User message: {user_question}, User ID: {user_id}, Message time: {message_time}, Username: {username}, First name: {first_name}, Last name: {last_name}, Chat ID: {chat_id}, Message ID: {message_id}, Is bot: {is_bot}, Language code: {language_code}")

    #  ----------------------------------------  AI Logic starts here ----------------------------------------  #
    normalized_message = normalize_text(user_question)

    input_text = build_input_output([normalized_message])
    
    model_answer = generate_response(input_text)
    await update.message.reply_text(model_answer)
    
    chat_info.append([normalized_message, model_answer, message_time, user_id, username, first_name, last_name, chat_id, message_id, is_bot, language_code])
    #  ----------------------------------------  AI Logic ends here ----------------------------------------  #
    # TODO: If chat_info is longer than 10, append it do dataset

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # TODO: Maybe implement a way to handle photos? 
    await update.message.reply_text("Sorry, I am not able to process images or files right now! Please, send me a text message.")


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(f"Exception while handling an update: {context.error}")

    if update is not None and update.message is not None:
        await update.message.reply_text("An error occurred. Please try again later.")
    else:
        print("Error occurred but update or update.message is None. Terminating active session.")

    # Terminate the active session to avoid having multiple active sessions on the Telegram API.
    await context.application.stop()

async def about(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hello! I am a personal GPT chatbot. I am trained to accurately replicate the conversation style of the person I was trained on. Send me a message, and let's chat!")
    

if __name__ == "__main__":
    print("Starting the bot...")
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    tokenizer = 0
    base_model = 0 
    model = 0
    

    print("Model loaded successfully.")


    app = ApplicationBuilder().token(token=TOKEN).build()

    # Start the bot
    app.add_handler(CommandHandler("start", about)) # TODO: CHANGE THE BOT COMAMND IN TELEGRAM FOR ABOUT    

    # Message handler
    chat_info = [] # [question, model_answer, message_time] 0: user_message, 1: model_answer, 2: message_time
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Handle any other message types
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    # Error
    app.add_error_handler(error)

    # Run the bot 
    print("Bot started.")
    app.run_polling(poll_interval=3)


