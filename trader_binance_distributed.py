from datetime import datetime, timedelta
import time
from binance.client import Client
import csv
import logging
import os
import settings
import data_manager
from policy_learner import PolicyLearner
import numpy as np
import ccxt
import win32api

#매번 조정해줘야할 변수들
isBuy = False                    #True는 매수포지션, False는 매도포지션
learned_coins = ["ETH"]
model_ver = 'policy_20190102215607'
#./having_coins/having_coins.txt 에 현재 보유중인 코인 목록 기록
#보유중인 코인 목록을 가져온다
having_coin_names = []
filepath = os.path.join('./having_coins', "having_coins.txt")
f = open(filepath,'r')
lines = f.readlines()
for line in lines:
    coin = line.rstrip('\n')
    having_coin_names.append(coin)
f.close()
having_coin_name = ""


# 9시가 될때까지 대기
while(1):
    now = datetime.now()
    time.sleep(1)
    if now.hour == 9:
        break;

buy_probs = []
sell_probs = []
buy_prob = 0
sell_prob = 0

#prev_date/prev_day는 오늘의 바로 전날을 의미(새벽에 켜두고 오전 9시에 자동실행 되게 해놓았을 시에)
#엑셀에서 data 읽어올때 사용하는 날짜 생성
now = datetime.now()
prev_day = now + timedelta(days=-1)
start_day = prev_day + timedelta(days=-119)
prev_day = prev_day.strftime('%Y-%m-%d')
start_day = start_day.strftime('%Y-%m-%d')
#binance에서 data 읽어올때 사용하는 날짜 생성
now = datetime.now()
prev_date = now + timedelta(days=-1)
prev_date = prev_date.strftime('%d %b, %Y')

#binance 접속
client = Client('', '')
binance = ccxt.binance({
    'apiKey': '',
    'secret': '',
    'options':{'adjustForTimeDifference': True }
    })
'''
#시간 싱크 맞춰줌(가끔 시간 안맞을때 실행, 관리자권한으로 실행 필요)
gt = client.get_server_time()
tt=time.gmtime(int((gt["serverTime"])/1000))
win32api.SetSystemTime(tt[0],tt[1],0,tt[2],tt[3],tt[4],tt[5],0)
'''

# 각 코인별 data를 최신 날짜로 업데이트
for coin in learned_coins:
    klines = client.get_historical_klines('{}BTC'.format(coin), Client.KLINE_INTERVAL_1DAY, "17 Aug, 2017", prev_date)
    with open('./data/chart_data/{}BTC.csv'.format(coin), 'w') as outcsv:
        writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        #writer.writerow(['date', 'open', 'high', 'low', 'close', 'volume'])
        for item in klines:
            #Write item to outcsv
            item[0] = int(item[0] / 1000)
            item[0] = datetime.utcfromtimestamp(item[0]).strftime('%Y-%m-%d')
            writer.writerow([item[0], item[1], item[2], item[3], item[4], item[5]])


if isBuy == True: #현재 학습된 코인을 갖고 있지 않다면(매수 포지션)
    #학습된 코인들 중 매수 확률이 0.5보다 높은 코인들을 분할매수
    isFirst = True
    for coin in learned_coins:
        #투자 시뮬레이션 파트, 각 코인별 매수확률을 구해서 buy_probs 리스트에 저장(index가 learned_coins와 동일하게)
        model_code = 'integrated'
        stock_code = coin + 'BTC'
        # 데이터 준비
        chart_data = data_manager.load_chart_data(
            os.path.join(settings.BASE_DIR,
                         'data/chart_data/{}.csv'.format(stock_code)))
        training_data = data_manager.build_training_data(chart_data)
        # 기간 필터링
        training_data = training_data[(training_data['date'] >= start_day) &
                                      (training_data['date'] <= prev_day)]
        training_data = training_data.dropna()
        # 차트 데이터 분리
        features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
        chart_data = training_data[features_chart_data]
        # 학습 데이터 분리
        features_training_data = [
            'high_low_ratio', 'open_close_ratio',
            'high_open_ratio', 'low_open_ratio',
            'high_close_ratio', 'low_close_ratio',
            'close_lastclose_ratio', 'volume_lastvolume_ratio'
            ]
        training_data = training_data[features_training_data]
        # 비 학습 투자 시뮬레이션 시작
        if isFirst == True:
            policy_learner = PolicyLearner(
                stock_code=stock_code, chart_data=chart_data, training_data=training_data,
                lr=0.00000001, discount_factor=0, start_epsilon=0, num_past_input=119)
            isFirst = False
        else:
            policy_learner.stock_code = stock_code
            policy_learner.chart_data = chart_data
            policy_learner.training_data = training_data
        policy_learner.trade(balance=20000,
                             model_path=os.path.join(
                                 settings.BASE_DIR,
                                 'models/{}/model_{}.h5'.format(model_code, model_ver)))
        buy_probs.append(policy_learner.action[0])

    #분할 매수 파트
    buy_probs = np.array(buy_probs)
    num_buy = 0
    if np.max(buy_probs) > 0.5: #분할 매수하기(구입할 코인 개수를 구한다음, 각 코인별 동일하게 분할매수)
        for coin in learned_coins: #구입할 코인 개수 구하기
            if buy_probs[learned_coins.index(coin)] > 0.5:
                num_buy++
        for coin in learned_coins: #분할 매수 실행
            if buy_probs[learned_coins.index(coin)] > 0.5:
                depth = client.get_order_book(symbol='{}BTC'.format(coin))
                market_buy_price = float(depth['asks'][0][0])
                btc_balance = float(client.get_asset_balance(asset='BTC')['free'])
                order = binance.create_market_buy_order('{}/BTC'.format(coin), (btc_balance/market_buy_price)/num_buy)
                num_buy--
                print(order)

                #보유중인 코인 목록 임시저장
                having_coin_names.append(coin)

        #보유중인 코인 목록 파일에 저장
        filepath = os.path.join('./having_coins', "having_coins.txt")
        if not(os.path.isdir('./having_coins')):
            os.makedirs(os.path.join('./having_coins'))
        f = open(filepath,'w')
        f.write('\n'.join(having_coin_names))
    

else: #현재 학습된 코인을 갖고 있다면(매도 포지션)
    #학습된 코인들의 매도 확률을 계산하여, 0.5보다 높은건 매도 후, 다시 매수 포지션 잡기(매수확률 0.5보다 높은애들 매수)
    ##분할 매도 파트
    for having_coin_name in having_coin_names:
        model_code = 'integrated'
        stock_code = having_coin_name + 'BTC'
        # 데이터 준비
        chart_data = data_manager.load_chart_data(
            os.path.join(settings.BASE_DIR,
                         'data/chart_data/{}.csv'.format(stock_code)))
        training_data = data_manager.build_training_data(chart_data)
        # 기간 필터링
        training_data = training_data[(training_data['date'] >= start_day) &
                                      (training_data['date'] <= prev_day)]
        training_data = training_data.dropna()
        # 차트 데이터 분리
        features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
        chart_data = training_data[features_chart_data]
        # 학습 데이터 분리
        features_training_data = [
            'high_low_ratio', 'open_close_ratio',
            'high_open_ratio', 'low_open_ratio',
            'high_close_ratio', 'low_close_ratio',
            'close_lastclose_ratio', 'volume_lastvolume_ratio'
            ]
        training_data = training_data[features_training_data]
        # 비 학습 투자 시뮬레이션 시작
        policy_learner = PolicyLearner(
            stock_code=stock_code, chart_data=chart_data, training_data=training_data,
            lr=0.00000001, discount_factor=0, start_epsilon=0, num_past_input=119)
        policy_learner.trade(balance=20000,
                             model_path=os.path.join(
                                 settings.BASE_DIR,
                                 'models/{}/model_{}.h5'.format(model_code, model_ver)))
        sell_prob = policy_learner.action[1]

        #매도확률이 0.5보다 높은것들 매도하고, 보유코인 목록 리스트에서 삭제하기
        if sell_prob > 0.5:
            balance = float(client.get_asset_balance(asset=having_coin_name)['free'])
            order = binance.create_market_sell_order('{}/BTC'.format(having_coin_name), balance)
            print(order)
            having_coin_names.remove(having_coin_name)
    #보유코인 목록파일 업데이트
    filepath = os.path.join('./having_coins', "having_coins.txt")
    if not(os.path.isdir('./having_coins')):
        os.makedirs(os.path.join('./having_coins'))
    f = open(filepath,'w')
    f.write('\n'.join(having_coin_names))

    ##분할 매수 파트
    #더 높은 매수 확률의 코인을 매수하기 위해 매도후 매수 포지션 잡기
    for coin in learned_coins:
        model_code = 'integrated'
        stock_code = coin + 'BTC'
        # 데이터 준비
        chart_data = data_manager.load_chart_data(
            os.path.join(settings.BASE_DIR,
                            'data/chart_data/{}.csv'.format(stock_code)))
        training_data = data_manager.build_training_data(chart_data)
        # 기간 필터링
        training_data = training_data[(training_data['date'] >= start_day) &
                                      (training_data['date'] <= prev_day)]
        training_data = training_data.dropna()
        # 차트 데이터 분리
        features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
        chart_data = training_data[features_chart_data]

        # 학습 데이터 분리
        features_training_data = [
            'high_low_ratio', 'open_close_ratio',
            'high_open_ratio', 'low_open_ratio',
            'high_close_ratio', 'low_close_ratio',
            'close_lastclose_ratio', 'volume_lastvolume_ratio'
            ]
        training_data = training_data[features_training_data]
        # 비 학습 투자 시뮬레이션 시작
        policy_learner.stock_code = stock_code
        policy_learner.chart_data = chart_data
        policy_learner.training_data = training_data
        policy_learner.trade(balance=20000,
                             model_path=os.path.join(
                                    settings.BASE_DIR,
                                    'models/{}/model_{}.h5'.format(model_code, model_ver)))
        buy_probs.append(policy_learner.action[0])

    buy_probs = np.array(buy_probs)
    num_buy = 0
    if np.max(buy_probs) > 0.5: #매수할 코인이 있다면, 분할 매수하기(구입할 코인 개수를 구한다음, 각 코인별 동일하게 분할매수)
        for coin in learned_coins: #구입할 코인 개수 구하기
            if buy_probs[learned_coins.index(coin)] > 0.5:
                num_buy++
        for coin in learned_coins: #분할 매수 실행
            if buy_probs[learned_coins.index(coin)] > 0.5:
                depth = client.get_order_book(symbol='{}BTC'.format(coin))
                market_buy_price = float(depth['asks'][0][0])
                btc_balance = float(client.get_asset_balance(asset='BTC')['free'])
                order = binance.create_market_buy_order('{}/BTC'.format(coin), (btc_balance/market_buy_price)/num_buy)
                num_buy--
                print(order)
                #보유중인 코인 목록 임시저장
                having_coin_names.append(coin)
        #보유중인 코인 목록 파일에 저장
        filepath = os.path.join('./having_coins', "having_coins.txt")
        if not(os.path.isdir('./having_coins')):
            os.makedirs(os.path.join('./having_coins'))
        f = open(filepath,'w')
        f.write('\n'.join(having_coin_names))
