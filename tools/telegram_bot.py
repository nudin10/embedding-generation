import os
from telegram.ext import Application
import asyncio
from tools.logger import Logger
import logging

class TelegramBot:
    def __init__(self, debug=False):
        self.logger = Logger(name="TelegramBot", level=logging.DEBUG if debug else logging.INFO)
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if self.token == "":
            self.token = os.getenv("RUNPOD_SECRET_TELEGRAM_BOT_TOKEN")

        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if self.chat_id == "":
            self.chat_id == os.getenv("RUNPOD_SECRET_TELEGRAM_CHAT_ID")

        self.app = Application.builder().token(self.token).build()

    async def send_message(self, message: str):
        try:
            await self.app.bot.send_message(chat_id=self.chat_id, text=message)
            await asyncio.sleep(1)
        except Exception as e:
            raise

bot = TelegramBot()

async def send_messages(messages: list[str]) -> None:
    for message in messages:
        try:
            await bot.send_message(message)
        except Exception as e:
            bot.logger.error(f"Error sending message: {e}")
            raise

async def send_message(message: str) -> None:
    try:
        await bot.send_message(message)
    except Exception as e:
        bot.logger.error(f"Error sending message: {e}")
        raise

async def send_error(message: str) -> None:
    try:
        await bot.send_message("[ERROR]: "+message)
    except Exception as e:
        bot.logger.error(f"Error sending message: {e}")
        raise

async def send_warning(message: str) -> None:
    try:
        await bot.send_message("[WARNING]: "+message)
    except Exception as e:
        bot.logger.error(f"Error sending message: {e}")
        raise
