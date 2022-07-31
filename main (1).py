import telebot
from functions import *
import io


welcome_message = """
Привет, я <b>бот</b>, который может находить похожие картинки

На данный момент в моей базе есть всего 5 классов изображений:
    Ромашка
    Тюльпан
    Роза
    Одуванчик
    Подсолнух
    
Ты можешь отправить мне фотографию любого из этих цветков, а я запросто найду 10 похожих изображений
"""
text_reply = "Простите, но я пока не умею общаться с людьми(:\nОтправьте мне картинку"

BOT_TOKEN = "Your bot token"

bot = telebot.TeleBot(BOT_TOKEN)


def extract_photo(message):
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    return Image.open(io.BytesIO(downloaded_file))


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(
        message.chat.id,
        welcome_message,
        parse_mode='HTML'
    )


@bot.message_handler(content_types=['photo'])
def reply_image(message):
    image = extract_photo(message)
    preprocessed_image = preprocess_image(image)
    img_vec = model.predict(preprocessed_image)
    indices = knn.kneighbors(img_vec, 10, return_distance=False)
    filenames = return_filenames(indices, filename='fnames.txt')
    reply = [telebot.types.InputMediaPhoto(open(fname, 'rb')) for fname in filenames]
    bot.send_media_group(message.chat.id, reply)


@bot.message_handler(content_types=['text'])
def reply_text(message):
    bot.send_message(message.chat.id, text_reply)


if __name__ == '__main__':
    model = createVGG19()

    vectors = create_vectors(model, directory='images')
    filenames = create_file_list(directory='images')

    knn = createNeighbors(vectors)

    bot.infinity_polling()

