import logging
import os
import settings
import data_manager
import numpy as np
from policy_learner import PolicyLearner
from keras.models import load_model
from binance.client import Client
from datetime import datetime, timedelta
import csv

if __name__ == '__main__':
    model_code = 'integrated'

    #매번 세팅 필요한 변수들
    learned_coins = ["ETH"]
    new_coins = [""]
    model_ver = '20190102215607' # model to load
    #★학습 시작,끝 날짜(datetime(연,월,일)) 매번 세팅 필요
    #가장 마지막 일자는 학습 못함(다음날의 reward를 구할수 없기때문)
    #최소한 마지막 2일간을 입력해줘야함
    start_day = datetime(2019,1,1,9) + timedelta(days=-119) #기존코인 시작일
    prev_day_temp = datetime(2019,1,2,9) #기존코인 + 새코인 끝 날짜

    policy_model_ver = 'policy_{}'.format(model_ver)
    value_model_ver = 'value_{}'.format(model_ver)
    timestr = settings.get_time_str()

    #코인 데이터 다운로드(from binance)
    #prev_date/prev_day는 오늘의 바로 전날을 의미(새벽에 켜두고 오전 9시에 자동실행 되게 해놓았을 시에)
    #엑셀에서 data 읽어올때 사용하는 날짜 생성
    prev_day = prev_day_temp.strftime('%Y-%m-%d')
    start_day = start_day.strftime('%Y-%m-%d')
    #binance에서 data 읽어올때 사용하는 날짜 생성
    prev_date = prev_day_temp
    prev_date = prev_date.strftime('%d %b, %Y')

    #binance 접속
    client = Client('', '')
    
    # 코인 데이터 준비
    first = True
    learned_coins.extend(new_coins)
    for stock_code in learned_coins:
        # 각 코인별 data를 최신 날짜로 업데이트
        klines = client.get_historical_klines('{}BTC'.format(stock_code), Client.KLINE_INTERVAL_1DAY, "17 Aug, 2017", prev_date)
        with open('./data/chart_data/{}BTC.csv'.format(stock_code), 'w') as outcsv:
            writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            for item in klines:
                #Write item to outcsv
                item[0] = int(item[0] / 1000)
                item[0] = datetime.utcfromtimestamp(item[0]).strftime('%Y-%m-%d')
                writer.writerow([item[0], item[1], item[2], item[3], item[4], item[5]])

        chart_data = data_manager.load_chart_data(
            os.path.join(settings.BASE_DIR,
                         'data/chart_data/{}BTC.csv'.format(stock_code)))
        training_data = data_manager.build_training_data(chart_data)

        # 기간 필터링
        if stock_code in new_coins:
            training_data = training_data[(training_data['date'] >= '2013-12-27') &
                                          (training_data['date'] <= prev_day)]
        else:
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

        if first is True:
            # 강화학습 시작
            policy_learner = PolicyLearner(
                stock_code=stock_code, chart_data=chart_data, training_data=training_data,
                policy_model_path=os.path.join(settings.BASE_DIR,'models/{}/model_{}.h5'.format(model_code, policy_model_ver)),
                value_model_path=os.path.join(settings.BASE_DIR,'models/{}/model_{}.h5'.format(model_code, value_model_ver)),
                lr=0.00000001, discount_factor=0, start_epsilon=0, num_past_input=119,
                load_weight_and_learn=True)
            first = False
        else:
            policy_learner.stock_code = stock_code
            policy_learner.chart_data = chart_data
            policy_learner.training_data = training_data

        # 여기가 핵심
        policy_learner.fit(balance=1, num_epoches=1000)


    # 정책 신경망을 파일로 저장
    model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % model_code)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model_policy_%s.h5' % timestr)
    policy_learner.policy_network.save(model_path,include_optimizer=False, overwrite=True)
    #policy_learner.policy_network_obj.save_model(model_path)
    model_path = os.path.join(model_dir, 'model_value_%s.h5' % timestr)
    policy_learner.value_network.save(model_path,include_optimizer=False, overwrite=True)
    #policy_learner.value_network_obj.save_weights(model_path)
