# coding=utf-8
import pandas as pd
import numpy as np
import telegram
from sklearn.neighbors import KDTree
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import time
import os
import datetime

TELEGRAM_TOKEN = '521985001:AAFjybE5ZOIlxdSzozxjPa0Gx0lU1EqIzC4'
MAPS_TOKEN = 'AIzaSyCwkxYSmXosfXb6_YtmbE5USz8OigOHjws'
MAPS_URL = 'https://maps.googleapis.com/maps/api/staticmap?'
CSV_FILE_URL = 'https://data.buenosaires.gob.ar/api/files/cajeros-automaticos.csv/download/csv'
CSV_FILE_LOCAL = "./cajeros-automaticos-procesados.csv"
SUPPLY_DATE_TXT = "ultimo-reabastecimiento.txt"
N_ATM = 3
MIN_ATM_BANK = 1  # min atm number per bank, dataset have some row with 0 atms
MAX_DIST = 0.5  # in km
N_DRAWS = 10  # number of draws from atm before saving into csv.


class TelegramBot(object):

    def __init__(self):
        startTime = time.time()
        self.updater = Updater(token=TELEGRAM_TOKEN)
        self.dispatcher = self.updater.dispatcher
        self.atm_locator = AtmLocator()

        self.dispatcher.add_handler(CommandHandler('start', self.__start))
        self.dispatcher.add_handler(CommandHandler('link', self.__command))
        self.dispatcher.add_handler(CommandHandler('banelco', self.__command))
        self.dispatcher.add_handler(MessageHandler(Filters.location, self.__location))
        endTime = time.time()
        print("init time: " + str(endTime - startTime))

    def __start(self, bot, update):
        bot.send_message(chat_id=update.message.chat_id,
                         text="Hola!!! Soy el bot más inteligete de Banjampp!")

    def __command(self, bot, update):
        atm_type = update.message.text.replace('/', '')
        text = "Hola! para enviarte los cajeros de la red {} debo solicitar tu ubicación".format(atm_type.upper())

        keyboard = telegram.KeyboardButton("Oprime para confirmar!",
                                           request_contact=None,
                                           request_location=True)
        reply_markup = telegram.ReplyKeyboardMarkup([[keyboard]])
        bot.send_message(chat_id=update.message.chat_id,
                         text=text,
                         reply_markup=reply_markup)

    def __location(self, bot, update):
        startTime = time.time()
        user_lat = update.message.location.latitude
        user_lon = update.message.location.longitude
        atm_types = ['BANELCO', 'LINK']
        atm_type = ""

        print("Request from {} with chat_id: {}".format(update.message.chat.first_name, update.message.chat.id))

        for at in atm_types:
            if at in update.message.reply_to_message.text.upper():
                atm_type = at
                break

        df = self.atm_locator.lookup(user_lat, user_lon, atm_type)

        bot.send_message(chat_id=update.message.chat_id,
                         text=self.__generate_resp_msg(df))
        bot.send_message(chat_id=update.message.chat_id,
                         text=self.__generate_static_map(user_lat, user_lon, df))

        endTime = time.time()
        print("Request response time:" + str(endTime - startTime))

    def __generate_resp_msg(self, df):
        resp_msg = ""
        i = 1
        msg = "{})\nEl Banco: {} posee {} teminal(es) de la red {} con un aprox. de {} retiros disponibles y se encuentra ubicado en {}\n\n"

        if df is not None:
            for index, row in df.iterrows():
                resp_msg += msg.format(str(i), row['BANCO'], str(row['TERMINALES']), row['RED'], str(row['RECARGAS']), row['DOM_GEO'])
                i += 1
        else:
            resp_msg = "No se encontró cajeros cercanos a su ubicación, gracias por usar el servicio Banjampp."

        return resp_msg

    def __generate_static_map(self, user_lat, user_lon, df):
        center = '{},{}'.format(str(user_lat), str(user_lon))
        size = '500x400'
        zoom = '15'
        marker = '&markers=color:{}|label:{}|{}'
        i = 1
        params = 'center={}&size={}&zoom={}&key={}'.format(center, size, zoom, MAPS_TOKEN)
        params += marker.format("blue", "V", center)

        if df is not None:
            for index, row in df.iterrows():
                params += marker.format("red", str(i), "{},{}".format(row['LAT'], row['LNG']))
                i += 1

        return MAPS_URL + params

    def start_bot(self):
        self.updater.start_polling()


class AtmLocator(object):

    def __init__(self):
        self.fh = FileHandler()
        self.df = self.fh.read_file()
        self.last_supply = self.fh.read_atm_supply_date()
        # Create a 3D-Tree with RED_CODE as 1 if RED values are 'BANELCO' and 0 if they're 'LINK'
        self.tree = KDTree(np.array(self.df[['LAT', 'LNG', 'RED_CODE']]), leaf_size=3)

    def lookup(self, user_lat, user_lon, atm_type):

        df_temp = self.__distance_calc(user_lat, user_lon, self.__query(user_lat, user_lon, atm_type))
        # If no atm is close to user location return None
        if df_temp.index.size <= 0:
            return None
        else:
            self.__suply_atm(df_temp)
            self.__money_draw(df_temp.index.values)
            return df_temp

    def __query(self, user_lat, user_lon, atm_type):
        user_location = np.array([[user_lat,
                                   user_lon,
                                   1 if atm_type == "BANELCO" else 0]])

        dist, ind = self.tree.query(user_location, k=N_ATM)

        df = pd.DataFrame(self.df,
                          index=ind[0],
                          columns=['ID', 'LAT', 'LNG', 'BANCO', 'RED', 'DOM_GEO', 'TERMINALES', 'BARRIO', 'RETIROS', 'RECARGAS'])

        return df[df['RECARGAS'] > 0]

    def __distance_calc(self, user_lat, user_lon, df_temp):

        def haversine_np(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))
            km = 6367 * c

            return km

        df_temp['ORIGIN_LAT'] = user_lat
        df_temp['ORIGIN_LNG'] = user_lon
        df_temp['DIST'] = haversine_np(df_temp['ORIGIN_LAT'].as_matrix(),
                                       df_temp['ORIGIN_LNG'].as_matrix(),
                                       df_temp['LAT'].as_matrix(),
                                       df_temp['LNG'].as_matrix())

        return df_temp[df_temp['DIST'] <= MAX_DIST]

    def __suply_atm(self, df_temp):
        now = datetime.datetime.today()
        now8am = now.replace(hour=8, minute=0, second=0, microsecond=0)
        # if there is more than 1 day of diff and day isn't satuday (5) or sunday  (6) supply atms
        if ((now - self.last_supply).days > 0 and now.weekday not in [5, 6]):
            self.df.loc[:, 'RECARGAS'] = self.df['TERMINALES'] * 1000
            df_temp.loc[:, 'RECARGAS'] = df_temp['TERMINALES'] * 1000
            self.last_supply = now8am
            self.fh.write_supply_date(now8am)

    def __money_draw(self, ind):
        p = [0.7, 0.2, 0.1]
        values = [1, 0, 0]
        actual_p = []
        actual_values = []
        i = 0

        while i < ind.size:
            actual_values.append(values[i])

            """if there is only one atm close, prob of drawing money from it is 1
               if there is two atm, prob of each is [0.75, 0.25]
               if there is three atm, prob of each is [0.7, 0.2, 0.1]"""
            if ind.size == 1:
                actual_p.append(1)
            elif ind.size == 2:
                actual_p.append(p[i] + 0.05)
            else:
                actual_p.append(p[i])
            i += 1

        draw_array = np.random.choice(np.array(actual_values), ind.size, p=actual_p, replace=False)
        self.df.at[ind, 'RETIROS'] = self.df.loc[ind, 'RETIROS'] + draw_array
        self.df.loc[ind, 'RECARGAS'] = self.df.loc[ind, 'RECARGAS'] - draw_array
        self.fh.write_file(self.df)


class FileHandler(object):

    def __init__(self):
        self.n_writes = 0

    def read_file(self):
        # local file exists
        if os.path.exists(CSV_FILE_LOCAL):
            df = pd.read_csv(CSV_FILE_LOCAL, sep=';')
        # if not we read from URL and "clean" the data
        else:
            df = pd.read_csv(CSV_FILE_URL, sep=';')
            df = df.filter(items=['ID', 'LAT', 'LNG', 'BANCO', 'RED', 'DOM_GEO', 'TERMINALES', 'BARRIO'])
            df = df.dropna(axis=0, how='any')
            # change float 0,0 notation to 0.0
            df.loc[:, 'LAT'] = df['LAT'].apply(lambda s: float(s.replace(',', '.')))
            df.loc[:, 'LNG'] = df['LNG'].apply(lambda s: float(s.replace(',', '.')))
            # create new column with RED_CODE, used to build 3D KD-Tree
            df.loc[:, 'RED_CODE'] = df['RED'].apply(lambda red: 1 if red == "BANELCO" else 0)
            df.loc[:, 'RETIROS'] = 0
            # we asume that each LINK atm location has two terminals
            df.loc[df.index[df['RED'] == 'LINK'], 'TERMINALES'] = 2
            # this line remove locations with 0 atm, but LINK atm all have 0 from dataset
            # dfFilteredNoNan = dfFilteredNoNan[dfFilteredNoNan['TERMINALES'] >= MIN_ATM_BANK]
            df.loc[:, 'RECARGAS'] = df['TERMINALES'] * 1000

            self.write_file(df)  # save cleaned file immediately

        return df

    def write_file(self, df):
        if (self.n_writes % N_DRAWS == 0):
            df.to_csv(CSV_FILE_LOCAL, sep=";", index=False)
            print("LOCAL CSV FILE CREATED!!")
        self.n_writes += 1

    def read_atm_supply_date(self):
        now = datetime.datetime.today()
        # change day to yesterday in case there is no supply_date file
        date = now.replace(day=(now.day - 1), hour=8, minute=0, second=0, microsecond=0)
        if os.path.exists(SUPPLY_DATE_TXT):
            with open(SUPPLY_DATE_TXT) as file:
                date_string = file.read()
                date = datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        return date

    def write_supply_date(self, supply_date):
        with open(SUPPLY_DATE_TXT, "w") as file:
            file.write("%s" % supply_date)
            print("SUPPLY DATE FILE UPDATED!!")


if __name__ == '__main__':
    tb = TelegramBot()
    tb.start_bot()
    print("Running Bot...")
