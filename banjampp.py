# coding=utf-8
import logging
import pandas as pd
import numpy as np
import telegram
from sklearn.neighbors import KDTree
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters
import time
import sys

TELEGRAM_TOKEN = '521985001:AAFjybE5ZOIlxdSzozxjPa0Gx0lU1EqIzC4'
MAPS_TOKEN = 'AIzaSyCwkxYSmXosfXb6_YtmbE5USz8OigOHjws'
MAPS_URL = 'https://maps.googleapis.com/maps/api/staticmap?'
CSV_FILE = 'https://data.buenosaires.gob.ar/api/files/cajeros-automaticos.csv/download/csv'
# CSV_FILE = 'cajeros-automaticos.csv'
N_ATM = 3
MIN_ATM_BANK = 1  # min atm number per bank, dataset have some row with 0 atms
MAX_DIST = 0.5  # in km


class TelegramBot(object):

    def __init__(self):
        startTime = time.time()
        self.updater = Updater(token=TELEGRAM_TOKEN)
        self.dispatcher = self.updater.dispatcher
        self.atm_locator = AtmLocator()

        self.dispatcher.add_handler(CommandHandler('start', self.start))
        self.dispatcher.add_handler(CommandHandler('link', self.command))
        self.dispatcher.add_handler(CommandHandler('banelco', self.command))
        self.dispatcher.add_handler(MessageHandler(Filters.location, self.location))
        endTime = time.time()
        print("init time: " + str(startTime - endTime))

    def start(self, bot, update):
        bot.send_message(chat_id=update.message.chat_id, text="Hola!!! Soy el bot más inteligete de Banjampp!")

    def command(self, bot, update):
        atm_type = update.message.text.replace('/', '')
        text = "Hola! para enviarte los cajeros de la red {} debo solicitar tu ubicación".format(atm_type)
        keyboard = telegram.KeyboardButton("Oprime para confirmar!", request_contact=None, request_location=True)
        reply_markup = telegram.ReplyKeyboardMarkup([[keyboard]])
        bot.send_message(chat_id=update.message.chat_id, text=text,
                         reply_markup=reply_markup)

    def location(self, bot, update):
        startTime = time.time()
        user_lat = update.message.location.latitude
        user_lon = update.message.location.longitude
        atm_types = ['banelco', 'link']
        atm_type = ""
        for at in atm_types:
            if at in update.message.reply_to_message.text:
                atm_type = at
                break
        df = self.atm_locator.lookup(user_lat, user_lon, atm_type)
        bot.send_message(chat_id=update.message.chat_id, text=self.generate_resp_msg(df))
        bot.send_message(chat_id=update.message.chat_id, text=self.generate_static_map(user_lat, user_lon, df))
        endTime = time.time()
        print("response time: " + str(endTime - startTime))

    # bien -0.0006422996520996094
    def generate_resp_msg(self, df):
        resp_msg = ""
        i = 1
        msg = "{})\nEl Banco: {} posee {} teminal(es) de la red {} y se encuentra ubicado en {}\n\n"

        for index, row in df.iterrows():
            resp_msg += msg.format(str(i), row['BANCO'], str(row['TERMINALES']), row['RED'], row['DOM_GEO'])
            i += 1
        return resp_msg

    # bien -0.0023131370544433594
    def generate_static_map(self, user_lat, user_lon, df):
        center = '{},{}'.format(str(user_lat), str(user_lon))
        size = '500x400'
        zoom = '15'
        marker = '&markers=color:{}|label:{}|{}'
        i = 1
        params = 'center={}&size={}&zoom={}&key={}'.format(center, size, zoom, MAPS_TOKEN)
        params += marker.format("blue", "V", center)

        for index, row in df.iterrows():
            params += marker.format("red", str(i), "{},{}".format(row['LAT'], row['LNG']))
            i += 1
        return MAPS_URL + params

    def start_bot(self):
        self.updater.start_polling()


class AtmLocator(object):

    def __init__(self):
        self.df = self.clean_dataset()
        # Create a 3D-Tree with RED_CODE as 1 if RED values are 'BANELCO' and 0 if they're 'LINK'
        self.tree = KDTree(np.array(self.df[['LAT', 'LNG', 'RED_CODE']]), leaf_size=3)
        self.history = {}
        self.buffer = []

    def clean_dataset(self):
        df = pd.read_csv(CSV_FILE, sep=';')
        dfFiltered = df.filter(items=['ID', 'LAT', 'LNG', 'BANCO', 'RED', 'DOM_GEO', 'TERMINALES', 'BARRIO'])
        dfFilteredNoNan = dfFiltered.dropna(axis=0, how='any')
        # change float 0,0 notation to 0.0
        dfFilteredNoNan.loc[:, 'LAT'] = dfFilteredNoNan['LAT'].apply(lambda s: float(s.replace(',', '.')))
        dfFilteredNoNan.loc[:, 'LNG'] = dfFilteredNoNan['LNG'].apply(lambda s:  float(s.replace(',', '.')))
        # create new column with RED_CODE, used to build 3D-Tree
        dfFilteredNoNan.loc[:, 'RED_CODE'] = dfFilteredNoNan['RED'].apply(lambda red: 1 if red == "BANELCO" else 0)
        dfFilteredNoNan.loc[:, 'DRAW'] = 0
        # this line remove locations with 0 atm
        # dfFilteredNoNan = dfFilteredNoNan[dfFilteredNoNan['TERMINALES'] >= MIN_ATM_BANK]

        return dfFilteredNoNan

    def lookup(self, user_lat, user_lon, atm_type):

        startTime = time.time()
        # user_location with 3 axis
        user_location = np.array([[user_lat, user_lon, 1 if atm_type == "banelco" else 0]])
        print(user_location)
        dist, ind = self.tree.query(user_location, k=N_ATM)
        endTime = time.time()
        print("TIME looking KDTree: " + str(endTime - startTime))
        startTime = time.time()
        df_temp = pd.DataFrame(self.df, index=ind[0], columns=['ID', 'LAT', 'LNG', 'BANCO', 'RED', 'DOM_GEO', 'TERMINALES', 'BARRIO', 'DRAW'])
        endTime = time.time()
        print("time creating df_temp: " + str(endTime - startTime))

        startTime = time.time()
        df_temp['ORIGIN_LAT'] = user_lat
        df_temp['ORIGIN_LNG'] = user_lon
        df_temp['DIST'] = self.haversine_np(df_temp['ORIGIN_LAT'].as_matrix(), 
                                            df_temp['ORIGIN_LNG'].as_matrix(), 
                                            df_temp['LAT'].as_matrix(), 
                                            df_temp['LNG'].as_matrix())
        df_temp = df_temp[df_temp['DIST'] <= MAX_DIST]
        #df_temp['RECARGAS'] = self.add_draw(df_temp.index.values, df_temp['TERMINALES'])
        # self.df.set_value(ind[0],self.df['DRAW'],[[0.7, 0.2, 0.1]])
        draw_array = np.random.choice(np.array([1, 0, 0]), 3, p=[0.7, 0.2, 0.1], replace=False)
        self.df.at[ind[0], 'DRAW'] = self.df.loc[ind[0], 'DRAW'] + draw_array
        print("------------------")
        example = self.df.loc[ind[0], ['TERMINALES']] * 1000
        print(self.df.loc[ind[0], ['TERMINALES', 'DRAW']])
        print(example.loc[ind[0], 'TERMINALES'] - self.df.loc[ind[0], 'DRAW'])
        df_temp1 = pd.DataFrame(self.df, index=ind[0], columns=['ID', 'LAT', 'LNG', 'BANCO', 'RED', 'DOM_GEO', 'TERMINALES', 'BARRIO', 'DRAW'])
        #print(df_temp1)
        endTime = time.time()
        #print("time for haversine_np:" + str(endTime - startTime))
        return df_temp

    def haversine_np(self, lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))
            km = 6367 * c
            return km

    def add_draw(self, index, number_atm):
        print(index)
        print(number_atm)
        percent = np.array([[0.7, 0.2, 0.1]])
        dictionary = dict(zip(index, percent))
        number_a = number_atm * 1000
        print(number_a - number_atm)
        for key, value in dictionary:
            if (self.history.get(key)):
                self.history[key] += value
            else:
                self.history[key] = value
        #self.add_buffer(index)
        return number_atm

if __name__ == '__main__':
    tb = TelegramBot()
    tb.start_bot()
