# coding=utf-8
import logging
import pandas as pd
import numpy as np
import telegram
from sklearn.neighbors import KDTree
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters

TELEGRAM_TOKEN = '521985001:AAFjybE5ZOIlxdSzozxjPa0Gx0lU1EqIzC4'
MAPS_TOKEN = 'AIzaSyCwkxYSmXosfXb6_YtmbE5USz8OigOHjws'
MAPS_URL = 'https://maps.googleapis.com/maps/api/staticmap?'
CSV_FILE = 'https://data.buenosaires.gob.ar/api/files/cajeros-automaticos.csv/download/csv'
N_ATM = 3
MIN_ATM_BANK = 1  # min atm number per bank, dataset have some row with 0 atms
MAX_DIST = 0.5  # in km


class TelegramBot(object):

    def __init__(self):
        self.updater = Updater(token=TELEGRAM_TOKEN)
        self.dispatcher = self.updater.dispatcher
        self.n_atm = N_ATM
        self.atm_locator = AtmLocator()

        self.dispatcher.add_handler(CommandHandler('start', self.start))
        self.dispatcher.add_handler(CommandHandler('link', self.command))
        self.dispatcher.add_handler(CommandHandler('banelco', self.command))
        self.dispatcher.add_handler(MessageHandler(Filters.location, self.location))

    def start(self, bot, update):
        bot.send_message(chat_id=update.message.chat_id, text="Hola!!! Soy el bot más inteligete de Banjampp!")

    def link(self, bot, update):
        bot.send_message(chat_id=update.message.chat_id, text="Listando cajeros link")

    def command(self, bot, update):
        atm_type = update.message.text.replace('/', '')
        text = "Hola! para enviarte los cajeros de la red {} debo solicitar tu ubicación".format(atm_type)
        keyboard = telegram.KeyboardButton("Oprime para confirmar!", request_contact=None, request_location=True)
        reply_markup = telegram.ReplyKeyboardMarkup([[keyboard]])
        bot.send_message(chat_id=update.message.chat_id, text=text,
                         reply_markup=reply_markup)

    def location(self, bot, update):
        atm_types = ['banelco', 'link']
        atm_type = ""
        for at in atm_types:
            if at in update.message.reply_to_message.text:
                atm_type = at
                break
        user_lat = update.message.location.latitude
        user_lon = update.message.location.longitude
        df = self.atm_locator.lookup(user_lat, user_lon, atm_type)
        bot.send_message(chat_id=update.message.chat_id, text=self.generate_resp_msg(df))
        bot.send_message(chat_id=update.message.chat_id, text=self.generate_static_map(user_lat, user_lon, df))

    def generate_resp_msg(self, df):
        resp_msg = ""
        i = 1

        msg = "{}) El Banco: {} posee {} teminal(es) de la red {} y se encuentra ubicado en {}\n"

        for index, row in df.iterrows():
            resp_msg += msg.format(str(i), row['BANCO'], str(row['TERMINALES']), row['RED'], row['DOM_GEO'])
            i += 1
        return resp_msg

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
        self.dfLink = self.df[self.df['RED'] == "link".upper()]
        self.dfBanelco = self.df[self.df['RED'] == "banelco".upper()]
        self.treeBanelco = KDTree(self.dfBanelco[['LAT', 'LNG']], leaf_size=2)
        self.treeLink = KDTree(self.dfLink[['LAT', 'LNG']], leaf_size=2)

    def clean_dataset(self):
        df = pd.read_csv(CSV_FILE, sep=';')
        dfFiltered = df.filter(items=['ID', 'LAT', 'LNG', 'BANCO', 'RED', 'DOM_ORIG', 'DOM_GEO', 'TERMINALES', 'BARRIO', 'COMUNA'])
        dfFilteredNoNan = dfFiltered.dropna(axis=0, how='any')
        dfFilteredNoNan['LAT'] = dfFilteredNoNan['LAT'].apply(lambda s: s.replace(',', '.'))
        dfFilteredNoNan['LNG'] = dfFilteredNoNan['LNG'].apply(lambda s: s.replace(',', '.'))
        # dfFilteredNoNan = dfFilteredNoNan[dfFilteredNoNan['TERMINALES'] >= MIN_ATM_BANK] #this line remove locations with 0 atm
        return dfFilteredNoNan

    def lookup(self, user_lat, user_lon, atm_type):

        user_location = np.array([[user_lat, user_lon]])
        if atm_type == "banelco":
            print("banelco")
            dist, ind = self.treeBanelco.query(user_location, k=N_ATM)
        else:
            print("link")
            dist, ind = self.treeLink.query(user_location, k=N_ATM)

        print(ind)
        print(dist)
        df_temp = pd.DataFrame(self.dfLink[self.dfLink['RED']], index=ind[0], columns=['ID', 'LAT', 'LNG', 'BANCO', 'RED', 'DOM_GEO', 'TERMINALES', 'BARRIO'])
        print(df_temp)
        print("---------------------------------------------------")
        # df_temp = df_temp[df_temp['RED'] == atm_type.upper()]
        # print(df_temp)
        df_temp['ORIGIN_LAT'] = user_lat
        df_temp['ORIGIN_LNG'] = user_lon
        df_temp['LAT'] = pd.to_numeric(df_temp['LAT'], downcast='float')
        df_temp['LNG'] = pd.to_numeric(df_temp['LNG'], downcast='float')
        dest_lon = df_temp['LNG'].as_matrix()
        dest_lat = df_temp['LAT'].as_matrix()
        orig_lon = df_temp['ORIGIN_LNG'].as_matrix()
        orig_lat = df_temp['ORIGIN_LAT'].as_matrix()

        df_temp['DIST'] = self.haversine_np(user_lon, user_lat, dest_lon, dest_lat)
        df_temp = df_temp[df_temp['DIST'] <= MAX_DIST]
        return df_temp

    def haversine_np(self, lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

            dlon = lon2 - lon1
            dlat = lat2 - lat1

            a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

            c = 2 * np.arcsin(np.sqrt(a))
            km = 6367 * c
            return km


if __name__ == '__main__':
    tb = TelegramBot()
    tb.start_bot()
