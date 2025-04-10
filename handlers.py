import logging
from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import CommandStart
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from os import getenv
from dotenv import load_dotenv
from model import RagModel

load_dotenv()
TELEGRAM_TOKEN = getenv('TELEGRAM_TOKEN')

ragm = RagModel()

router = Router()

class StocksState(StatesGroup):
    situation_rag = State()
    situation_no_rag = State()

knopki = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text='Запрос с RAG'), 
               KeyboardButton(text='Запрос без RAG')]],
               resize_keyboard=True,
               input_field_placeholder='Нажмите кнопку'
)


@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer('Напишите свой запрос относительно модели Deepseek',
                         reply_markup=knopki)


@router.message(F.text == 'Запрос с RAG')
async def set_rag(message: Message, state: FSMContext):
    await state.set_state(StocksState.situation_rag)
    await message.answer('Задайте вопрос по статьям с Substack')

@router.message(F.text == 'Запрос без RAG')
async def set_no_rag(message: Message, state: FSMContext):
    await state.set_state(StocksState.situation_no_rag)
    await message.answer('Задайте вопрос')


@router.message(StocksState.situation_rag)
async def answer_rag(message: Message, state: FSMContext):
    await state.update_data(situation_rag=message.text)
    text = await state.get_data()
    await message.answer(f'Ваш вопрос: {text["situation_rag"]}')
    answer = ragm.ask_rag(text["situation_rag"]) 
    await message.answer(f'Вот что удалось найти: {answer}', reply_markup=knopki)
    await state.clear()

@router.message(StocksState.situation_no_rag)
async def answer_no_rag(message: Message, state: FSMContext):
    await state.update_data(situation_no_rag=message.text)
    text = await state.get_data()
    await message.answer(f'Ваш вопрос: {text["situation_no_rag"]}')
    answer = ragm.ask_no_rag(text["situation_no_rag"]) 
    await message.answer(f'Ответ модели: {answer}', reply_markup=knopki)
    await state.clear()


@router.message()
async def unknown_message(message: Message):
    await message.answer("Я вас не понимаю. Выберите кнопку.", reply_markup=knopki)

async def main():
    logging.basicConfig(level=logging.INFO)
    bot = Bot(token=TELEGRAM_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot, skip_updates=True)
