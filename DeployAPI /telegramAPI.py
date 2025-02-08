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

from helper_functions import change_prompts, normalize_text

with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    

TARGET_MAX_LENGTH = config['training_parameters']['TARGET_MAX_LENGTH']
OUTPUT_DIR = config['training_parameters']['OUTPUT_DIR']
MODEL_NAME = config['training_parameters']['MODEL_NAME']
TOKEN = config['personal_parameters']['TELEGRAM_API']
q_prompt, finetune_prompt, c_prompt, context_label = change_prompts(language="uk", df=None)


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

def build_input_output(chat_info: List) -> str:
    # TODO: Maybe implement a way to follow the time difference between messages
    print(chat_info)

    if chat_info[-1]["model_answer"]:
        for key, (question, answer) in enumerate(zip(chat_info[:][0], chat_info[:][0])):
            context += f" <Q{key + 1}> {question} <A{key + 1}> {answer}"

    print(f"Context: {context}")

    input_text = (
        f"[{q_prompt}]:{chat_info['user_message']}\n"
        f"[{c_prompt}]: {context}\n"
    )

    return input_text


async def start_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hello! I am a personal GPT chatbot. I am trained to accurately replicate the conversation style of the person I was trained on. Send me a message, and let's chat!")


async def start_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message_time = update.message.date
    message_type = update.message.chat.type
    user_question = update.message.text

    print(f"Message type: {message_type}, User message: {user_question}")

    #  ----------------------------------------  AI Logic starts here ----------------------------------------  #
    normalized_message = normalize_text(user_question)

    input_text = build_input_output(normalized_message, model_answer=model_answer, context=context)
    
    #model_answer = generate_response(input_text)
    model_answer = "This is a test message." + "I"
    await update.message.reply_text(model_answer)
    
    chat_info.append([normalized_message, model_answer, message_time])
    #  ----------------------------------------  AI Logic ends here ----------------------------------------  #

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Sorry, I didn't understand that command. Please type /start to begin the conversation.")
    

if __name__ == "__main__":
    print("Starting the bot...")
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    print("Model loaded successfully.")

    app = ApplicationBuilder().token(token=TOKEN).build()

    # Start the bot
    app.add_handler(CommandHandler("start", start_conversation))

    # Message handler
    chat_info = [] # [question, model_answer, message_time] 0: user_message, 1: model_answer, 2: message_time
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, start_conversation))

    # Error
    app.add_error_handler(error)

    # Run the bot 
    print("Polling...")
    app.run_polling(poll_interval=3)