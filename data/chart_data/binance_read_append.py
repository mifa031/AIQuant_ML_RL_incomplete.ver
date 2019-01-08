from binance.client import Client
import csv
from datetime import datetime

client = Client('', '')

klines = client.get_historical_klines("LTCBTC", Client.KLINE_INTERVAL_1DAY, "29 Dec, 2018", "29 Dec, 2018")

with open('LTCBTC.csv', 'a') as outcsv:   
    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    #writer.writerow(['date', 'open', 'high', 'low', 'close', 'volume'])
    for item in klines:
        #Write item to outcsv
        item[0] = int(item[0] / 1000)
        item[0] = datetime.utcfromtimestamp(item[0]).strftime('%Y-%m-%d')
        writer.writerow([item[0], item[1], item[2], item[3], item[4], item[5]])
