import logging
import pandas as pd
import numpy as np
import telegram
from sklearn.neighbors import KDTree
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters

TELEGRAM_TOKEN = '521985001:AAFjybE5ZOIlxdSzozxjPa0Gx0lU1EqIzC4'
CSV_FILE = 'https://data.buenosaires.gob.ar/api/files/cajeros-automaticos.csv/download/csv'
N_ATM = 3
MAX_DIST = 0.5 #in km


class TelegramBot(object):

    def __init__(self):
        self.updater = Updater(token=TELEGRAM_TOKEN)
        self.dispatcher = self.updater.dispatcher
        self.n_atm = N_ATM
        self.atm_locator = AtmLocator()

        self.dispatcher.add_handler(CommandHandler('start', self.start))
        self.dispatcher.add_handler(CommandHandler('link', self.link))
        self.dispatcher.add_handler(CommandHandler('banelco', self.banelco))
        self.dispatcher.add_handler(MessageHandler(Filters.text, self.echo))
        self.dispatcher.add_handler(MessageHandler(Filters.location, self.location))

    def start(self, bot, update):
        bot.send_message(chat_id=update.message.chat_id, text="Hola!!! Soy el bot más inteligete de Banjampp!")

    def link(self, bot, update):
        bot.send_message(chat_id=update.message.chat_id, text="Listando cajeros link")

    def banelco(self, bot, update):
        keyboard = telegram.KeyboardButton("Hola! para enviarte los cajeros de banelco debo solicitar tu ubicación",request_contact=None, request_location=True)
        reply_markup = telegram.ReplyKeyboardMarkup([[keyboard]])
        bot.send_message(chat_id=update.message.chat_id, text="Hola! para enviarte los cajeros de banelco debo solicitar tu ubicación",reply_markup=reply_markup)

    def echo(self, bot, update):
        bot.send_message(chat_id=update.message.chat_id, text=update.message.text)

    def location(self, bot, update):
        self.atm_locator.lookup(update.message.location.latitude, update.message.location.longitude)

    def start_bot(self):
        self.updater.start_polling()


class AtmLocator(object):

    def __init__(self):
        df = pd.read_csv(CSV_FILE, sep=';')
        dfFiltered = df.filter(items=['ID', 'LAT', 'LNG', 'BANCO', 'RED', 'DOM_ORIG', 'DOM_GEO', 'TERMINALES', 'BARRIO', 'COMUNA'])
        dfFilteredNoNan = dfFiltered.dropna(axis=0, how='any')
        dfFilteredNoNan['LAT'] = dfFilteredNoNan['LAT'].apply(lambda s: s.replace(',', '.'))
        dfFilteredNoNan['LNG'] = dfFilteredNoNan['LNG'].apply(lambda s: s.replace(',', '.'))
        self.df = dfFilteredNoNan
        #self.df['LAT'] = pd.to_numeric(self.df['LAT'], downcast='float')
        #self.df['LNG'] = pd.to_numeric(self.df['LNG'], downcast='float')
        X = np.array(self.df[['LAT', 'LNG']])
        self.tree = KDTree(X, leaf_size=2)

    def lookup(self, user_lat, user_lon):

        def haversine_np(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

            dlon = lon2 - lon1
            dlat = lat2 - lat1

            a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

            c = 2 * np.arcsin(np.sqrt(a))
            km = 6367 * c
            return km

        user_location = np.array([[user_lat, user_lon]])

        dist, ind = self.tree.query(user_location, k=N_ATM)
        df_temp = pd.DataFrame(self.df, index=ind[0], columns=['LAT', 'LNG', 'BANCO', 'DOM_GEO'])
        df_temp['ORIGIN_LAT'] = user_lat
        df_temp['ORIGIN_LNG'] = user_lon
        df_temp['LAT'] = pd.to_numeric(df_temp['LAT'], downcast='float')
        df_temp['LNG'] = pd.to_numeric(df_temp['LNG'], downcast='float')
        dest_lon = df_temp['LNG'].as_matrix()
        dest_lat = df_temp['LAT'].as_matrix()
        orig_lon = df_temp['ORIGIN_LNG'].as_matrix()
        orig_lat = df_temp['ORIGIN_LAT'].as_matrix()
        """print(type(dest_lon))
        print(type(dest_lat))
        print(type(orig_lon))
        print(type(orig_lat))"""
        """print(dest_lon)
        print(dest_lat)
        print(orig_lon)
        print(orig_lat)"""
        df_temp['dist'] = haversine_np(user_lon, user_lat, dest_lon, dest_lat)
        df_temp = df_temp[df_temp['dist'] <= MAX_DIST]
        print(df_temp)



if __name__ == '__main__':
    tb = TelegramBot()
    tb.start_bot()


    """df = pd.read_csv('https://data.buenosaires.gob.ar/api/files/cajeros-automaticos.csv/download/csv', sep=';')
    dfFiltered = df.filter(items=['ID', 'LAT', 'LNG', 'BANCO', 'RED', 'DOM_ORIG', 'DOM_GEO', 'TERMINALES', 'BARRIO', 'COMUNA'])
    dfFilteredNoNan = dfFiltered.dropna(axis=0, how='any')
    # print(dfFilteredNoNan.LAT)
    dfFilteredNoNan['LAT'] = dfFilteredNoNan['LAT'].apply(lambda s: s.replace(',', '.'))
    dfFilteredNoNan['LNG'] = dfFilteredNoNan['LNG'].apply(lambda s: s.replace(',', '.'))
    X = np.array(dfFilteredNoNan[['LAT', 'LNG']])
    user_location = np.array([[-34.598542, -58.428014]])
    user_location = np.array([[-34.598542, -58.428014]])
    # print(X)
    # a = "-23.2312313"
    # print(float(a))
    
    tree = KDTree(X, leaf_size=2)
    #print(X[0])
    dist, ind = tree.query(user_location, k=3)
    print(dist)
    print(ind)"""
