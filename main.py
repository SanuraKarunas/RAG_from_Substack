import asyncio
import logging
from aiogram import Bot, Dispatcher
from handlers import router
from dotenv import load_dotenv
from os import getenv

load_dotenv()
TELEGRAM_TOKEN = getenv('TELEGRAM_TOKEN')

async def main():
    logging.basicConfig(level=logging.INFO)
    bot = Bot(token=TELEGRAM_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot, skip_updates=True)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Бот выключен')