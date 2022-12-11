from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

TEXT_BUTTON_GET_RESULT_NST = "Медленно"
TEXT_BUTTON_GET_RESULT_FAST_NST = "Быстро"
TEXT_BUTTON_REPEAT = "Начать сначала"
TEXT_BUTTON_CHOICE_NST = "NST"
TEXT_BUTTON_CHOICE_AND_GET_RESULT_GAN = "GAN"

button_get_result_nst = KeyboardButton(TEXT_BUTTON_GET_RESULT_NST)
button_get_result_fast_nst = KeyboardButton(TEXT_BUTTON_GET_RESULT_FAST_NST)
button_repeat = KeyboardButton(TEXT_BUTTON_REPEAT)
button_choice_nst = KeyboardButton(TEXT_BUTTON_CHOICE_NST)
button_choice_and_get_result_gan = KeyboardButton(TEXT_BUTTON_CHOICE_AND_GET_RESULT_GAN)

markup_choice_model = ReplyKeyboardMarkup(
    resize_keyboard=True,
    one_time_keyboard=True
).add(
    button_choice_nst,
    button_choice_and_get_result_gan
)

markup_get_result_nst = ReplyKeyboardMarkup(
    resize_keyboard=True, one_time_keyboard=True
).row(
    button_get_result_nst,
    button_get_result_fast_nst
)

markup_repeat = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True).add(button_repeat)
