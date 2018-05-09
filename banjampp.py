import logging
import telegram
import numpy as np
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters
from sklearn.neighbors import KDTree
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

updater = Updater(token='521985001:AAFjybE5ZOIlxdSzozxjPa0Gx0lU1EqIzC4')

dispatcher = updater.dispatcher


def start(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="Hola!!! Soy el bot más inteligete de Banjampp!")


def link(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="Listando cajeros link")


def banelco(bot, update):
    keyboard = telegram.KeyboardButton("Hola! para enviarte los cajeros de banelco debo solicitar tu ubicación",request_contact=None, request_location=True)
    reply_markup = telegram.ReplyKeyboardMarkup([[keyboard]])
    # print(update)
    bot.send_message(chat_id=update.message.chat_id, text="Hola! para enviarte los cajeros de banelco debo solicitar tu ubicación",reply_markup=reply_markup)


start_handler = CommandHandler('start', start)
link_handler = CommandHandler('link', link)
banelco_handler = CommandHandler('banelco', banelco)


dispatcher.add_handler(start_handler)
dispatcher.add_handler(link_handler)
dispatcher.add_handler(banelco_handler)
updater.start_polling()


def echo(bot, update):
    # print(update)
    bot.send_message(chat_id=update.message.chat_id, text=update.message.text)


def location(bot, update):
    print(update.message.location)
    df = pd.read_csv('https://data.buenosaires.gob.ar/api/files/cajeros-automaticos.csv/download/csv', sep=';')
    dfFiltered = df.filter(items=['ID', 'LAT', 'LNG', 'BANCO', 'RED', 'DOM_ORIG', 'DOM_GEO', 'TERMINALES', 'BARRIO', 'COMUNA'])
    dfFilteredNoNan = dfFiltered.dropna(axis=0, how='any')
    # print(dfFilteredNoNan.LAT)
    dfFilteredNoNan['LAT'] = dfFilteredNoNan['LAT'].apply(lambda s: s.replace(',', '.'))
    dfFilteredNoNan['LNG'] = dfFilteredNoNan['LNG'].apply(lambda s: s.replace(',', '.'))
    X = np.array(dfFilteredNoNan[['LAT', 'LNG']])
    user_location= np.array([[update.message.location.latitude, update.message.location.longitude]])
    # print(X)
    # a = "-23.2312313"
    # print(float(a))
    
    tree = KDTree(X, leaf_size=2)
    print(X[0])
    dist, ind = tree.query(user_location, k=3)
    print(ind)
    print(dist)


echo_handler = MessageHandler(Filters.text, echo)
location_handler = MessageHandler(Filters.location, location)
dispatcher.add_handler(location_handler)
dispatcher.add_handler(echo_handler)

if __name__ == '__main__':
    print("hola")
    """df = pd.read_csv('https://data.buenosaires.gob.ar/api/files/cajeros-automaticos.csv/download/csv', sep=';')
    dfFiltered = df.filter(items=['ID', 'LAT', 'LNG', 'BANCO', 'RED', 'DOM_ORIG', 'DOM_GEO', 'TERMINALES', 'BARRIO', 'COMUNA'])
    dfFilteredNoNan = dfFiltered.dropna(axis=0, how='any')
    # print(dfFilteredNoNan.LAT)
    dfFilteredNoNan['LAT'] = dfFilteredNoNan['LAT'].apply(lambda s: s.replace(',', '.'))
    dfFilteredNoNan['LNG'] = dfFilteredNoNan['LAT'].apply(lambda s: s.replace(',', '.'))
    X = np.array(dfFilteredNoNan[['LAT', 'LNG']])

    # print(X)
    # a = "-23.2312313"
    # print(float(a))
    
    tree = KDTree(X, leaf_size=2)
    dist, ind = tree.query([X[0]], k=3)
    print(ind)"""
    #print(dist)

    """while(updater.running):
        update = updater.update_queue.get()
        if (update):
            print(update.message.text)"""

