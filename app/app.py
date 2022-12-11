import os
import logging
import logging.config
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from shutil import rmtree

import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext
import yaml

from .ml_model.fast_nst import FastNeuralStyleTransfer
from .ml_model.nst import NeuralStyleTransferModel
from .ml_model.cycle_gan import CycleGanModel
from .utils import ManagerModel
from .keyboards import (
    markup_choice_model,
    markup_get_result_nst,
    markup_repeat,
    TEXT_BUTTON_GET_RESULT_NST,
    TEXT_BUTTON_GET_RESULT_FAST_NST,
    TEXT_BUTTON_CHOICE_NST,
    TEXT_BUTTON_CHOICE_AND_GET_RESULT_GAN,
    TEXT_BUTTON_REPEAT
)

from config import (
    API_TOKEN,
    IMAGE_FILES_PATH,
    RESULT_IMG_NAME,
    CONTENT_IMG_NAME,
    STYLE_IMG_NAME
)

from .text_messages import messages

DEFAULT_LOGGING_CONFIG_FILEPATH = "logging.conf.yml"


def setup_logging():
    with open(DEFAULT_LOGGING_CONFIG_FILEPATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


setup_logging()
logger = logging.getLogger("style_transfer_bot")
bot = Bot(token=API_TOKEN)
loop = asyncio.get_event_loop()
dp = Dispatcher(bot, storage=MemoryStorage(), loop=loop)

POOL = ThreadPoolExecutor(max_workers=cpu_count())

MMODEL = ManagerModel({
    "nst": NeuralStyleTransferModel(),
    "fast_nst": FastNeuralStyleTransfer({'load_pretrained': True}),
    "cycle_gan": CycleGanModel()
})


class Form(StatesGroup):
    download_content = State()
    choice_model = State()
    download_style = State()
    result = State()


def get_path_to_session_img_dir(session_id: int) -> str:
    return IMAGE_FILES_PATH + str(session_id)


@dp.message_handler(commands=['start'])
@dp.message_handler(text=TEXT_BUTTON_REPEAT)
async def process_start_command(message: types.Message, state: FSMContext):
    logger.info(
        "User: (%s) with chat id: (%s) start session",
        message.from_user.username,
        message.chat.id
    )
    logger.debug("Current state: (%s)", await state.get_state())
    session_dir = get_path_to_session_img_dir(message.chat.id)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)

    await Form.download_content.set()
    logger.debug("Current state: (%s)", await state.get_state())

    await message.answer(messages["process_start_command"])


@dp.message_handler(state='*', commands=['help'])
async def process_help_command(message: types.Message):
    logger.info(
        "User: (%s) with chat id: (%s) called for help",
        message.from_user.username,
        message.chat.id
    )
    await message.reply(messages["process_help_command"])


async def download_image(message: types.Message, file_path: str):
    if "photo" in message:
        photo = message.photo.pop()
        await photo.download(file_path)
        logger.info(
            "User (%s) with chat id: (%s) successfully download content image(type: photo)",
            message.from_user.username,
            message.chat.id,
        )
        logger.debug("uploaded content image(type: photo): (%s) with path: (%s)", photo, file_path)
    elif "document" in message:
        document = message.document
        if document.mime_type.split("/")[0] == "image":
            await document.download(file_path)
            logger.info(
                "User (%s) with chat id: (%s) successfully download content image(type: document)",
                message.from_user.username,
                message.chat.id,
            )
            logger.debug("uploaded content image(type: document): (%s) with path: (%s)", document, file_path)


@dp.message_handler(
    content_types=[types.ContentType.PHOTO, types.ContentType.DOCUMENT],
    state=Form.download_content
)
async def download_content_image(message: types.Message, state: FSMContext):
    logger.info(
        "User: (%s) with chat id: (%s) start try download content image",
        message.from_user.username,
        message.chat.id
    )
    logger.debug("Current state: (%s)", await state.get_state())
    session_dir = get_path_to_session_img_dir(message.chat.id)
    await download_image(message, os.path.join(session_dir, CONTENT_IMG_NAME))
    await Form.choice_model.set()
    logger.debug("Current state: (%s)", await state.get_state())
    await message.answer(
        messages["download_content_image"],
        reply=False,
        reply_markup=markup_choice_model
    )


@dp.message_handler(
    state=Form.choice_model,
    text=TEXT_BUTTON_CHOICE_NST
)
async def pre_download_style_image(message: types.Message, state: FSMContext):
    logger.debug("Current state: (%s)", await state.get_state())
    await Form.download_style.set()
    logger.debug("Current state: (%s)", await state.get_state())
    await message.answer(
        messages["pre_download_style_image"],
    )


@dp.message_handler(
    state=Form.download_style,
    content_types=[types.ContentType.PHOTO, types.ContentType.DOCUMENT]
)
async def download_style_image(message: types.Message, state: FSMContext):
    logger.info(
        "User: (%s) with chat id: (%s) start try download style image",
        message.from_user.username,
        message.chat.id
    )
    logger.debug("Current state: (%s)", await state.get_state())
    session_dir = get_path_to_session_img_dir(message.chat.id)
    await download_image(message, os.path.join(session_dir, STYLE_IMG_NAME))
    await Form.result.set()
    logger.debug("Current state: (%s)", await state.get_state())
    await message.reply(
        messages["download_style_image"],
        reply_markup=markup_get_result_nst
    )


@dp.message_handler(
    state=Form.result,
    text=TEXT_BUTTON_GET_RESULT_FAST_NST
)
async def get_result_fast_nst(message: types.Message, state: FSMContext):
    logger.info(
        "User: (%s) with chat id: (%s) start try get result image from FastNST",
        message.from_user.username,
        message.chat.id
    )
    logger.debug("Current state: (%s)", await state.get_state())
    session_dir = get_path_to_session_img_dir(message.chat.id)
    await message.reply(messages["get_result_fast_nst"][0], reply=False)
    model = MMODEL.get_model_instance("fast_nst")
    _ = await dp.loop.run_in_executor(
        POOL,
        model.predict,
        os.path.join(session_dir, CONTENT_IMG_NAME),
        os.path.join(session_dir, STYLE_IMG_NAME),
        os.path.join(session_dir, RESULT_IMG_NAME)
    )
    result_img = types.InputFile(os.path.join(session_dir, RESULT_IMG_NAME))
    logger.info(
        "User: (%s) with chat id: (%s). Bot successfully construct result image (model: FastNST)",
        message.from_user.username,
        message.chat.id
    )
    logger.debug(
        "Bot successfully construct result image (model: FastNST) - (%s)",
        result_img
    )
    await bot.send_photo(message.from_user.id, result_img)
    rmtree(session_dir)
    await message.answer(
        messages["get_result_fast_nst"][1],
        reply=False,
        reply_markup=markup_repeat
    )
    await state.reset_state()
    logger.debug("Current state: (%s)", await state.get_state())


@dp.message_handler(
    state=Form.result,
    text=TEXT_BUTTON_GET_RESULT_NST
)
async def get_result_nst(message: types.Message, state: FSMContext):
    logger.info(
        "User: (%s) with chat id: (%s) start try get result image from NeuralStyleTransfer",
        message.from_user.username,
        message.chat.id
    )
    logger.debug("Current state: (%s)", await state.get_state())
    session_dir = get_path_to_session_img_dir(message.chat.id)
    await message.reply(
        messages["get_result_nst"][0],
        reply=False
    )
    model = MMODEL.get_model_instance("nst")
    _ = await dp.loop.run_in_executor(
        POOL,
        model.predict,
        os.path.join(session_dir, CONTENT_IMG_NAME),
        os.path.join(session_dir, STYLE_IMG_NAME),
        os.path.join(session_dir, RESULT_IMG_NAME)
    )
    result_img = types.InputFile(os.path.join(session_dir, RESULT_IMG_NAME))
    logger.info(
        "User: (%s) with chat id: (%s). Bot successfully construct result image (model: NeuralStyleTransfer)",
        message.from_user.username,
        message.chat.id
    )
    logger.debug(
        "Bot successfully construct result image (model: NeuralStyleTransfer) - (%s)",
        result_img
    )
    await bot.send_photo(message.from_user.id, result_img)
    rmtree(session_dir)
    await message.answer(
        messages["get_result_nst"][1],
        reply=False,
        reply_markup=markup_repeat
    )
    await state.reset_state()
    logger.debug("Current state: (%s)", await state.get_state())


@dp.message_handler(
    state=Form.choice_model or Form.result,
    text=TEXT_BUTTON_CHOICE_AND_GET_RESULT_GAN
)
async def get_result_cycle_gan(message: types.Message, state: FSMContext):
    logger.info(
        "User: (%s) with chat id: (%s) start try get result image from CycleGAN",
        message.from_user.username,
        message.chat.id
    )
    logger.debug("Current state: (%s)", await state.get_state())
    session_dir = get_path_to_session_img_dir(message.chat.id)
    await message.reply(
        messages["get_result_cycle_gan"][0],
        reply=False
    )
    model = MMODEL.get_model_instance("cycle_gan")
    _ = await dp.loop.run_in_executor(
        POOL,
        model.predict,
        os.path.join(session_dir, CONTENT_IMG_NAME),
        os.path.join(session_dir, RESULT_IMG_NAME)
    )
    result_img = types.InputFile(os.path.join(session_dir, RESULT_IMG_NAME))
    logger.info(
        "User: (%s) with chat id: (%s). Bot successfully construct result image (model: CycleGAN)",
        message.from_user.username,
        message.chat.id
    )
    logger.debug(
        "Bot successfully construct result image (model: CycleGAN) - (%s)",
        result_img
    )
    await bot.send_photo(message.from_user.id, result_img)
    rmtree(session_dir)
    await message.answer(
        messages["get_result_cycle_gan"][1],
        reply=False,
        reply_markup=markup_repeat
    )
    await state.reset_state()
    logger.debug("Current state: (%s)", await state.get_state())


@dp.message_handler(state='*', commands=['reset'])
async def reset_state_on_start(message: types.Message, state: FSMContext):
    logger.info(
        "User: (%s) with chat id: (%s) reset session",
        message.from_user.username,
        message.chat.id
    )
    await state.finish()
    session_dir = get_path_to_session_img_dir(message.chat.id)
    if os.path.exists(session_dir):
        rmtree(session_dir)

    logger.info(
        "User: (%s) with chat id: (%s) start session",
        message.from_user.username,
        message.chat.id
    )
    logger.debug("Current state: (%s)", await state.get_state())
    session_dir = get_path_to_session_img_dir(message.chat.id)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)

    await Form.download_content.set()
    logger.debug("Current state: (%s)", await state.get_state())

    await message.answer(messages["process_start_command"])
