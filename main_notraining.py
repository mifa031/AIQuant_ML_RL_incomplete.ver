import logging
import os
import settings
import data_manager
from policy_learner import PolicyLearner


if __name__ == '__main__':
    model_code = 'integrated'
    
    stock_code = 'XRPBTC'
    model_ver = 'policy_20181226141441'

    # 로그 기록
    #log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
    #timestr = settings.get_time_str()
    #file_handler = logging.FileHandler(filename=os.path.join(
    #    log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
    #stream_handler = logging.StreamHandler()
    #file_handler.setLevel(logging.DEBUG)
    #stream_handler.setLevel(logging.INFO)
    #logging.basicConfig(format="%(message)s",
    #    handlers=[file_handler, stream_handler], level=logging.DEBUG)

    # 주식 데이터 준비
    chart_data = data_manager.load_chart_data(
        os.path.join(settings.BASE_DIR,
                     'data/chart_data/{}.csv'.format(stock_code)))
    #prep_data = data_manager.preprocess(chart_data)
    #training_data = data_manager.build_training_data(prep_data)
    training_data = data_manager.build_training_data(chart_data)

    # 기간 필터링
    training_data = training_data[(training_data['date'] >= '2018-08-29') &
                                  (training_data['date'] <= '2018-12-26')]
    training_data = training_data.dropna()

    # 차트 데이터 분리
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
    chart_data = training_data[features_chart_data]

    # 학습 데이터 분리
    features_training_data = [
        'high_low_ratio', 'open_close_ratio',
        'high_open_ratio', 'low_open_ratio',
        'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        #'close_ma5_ratio', 'volume_ma5_ratio',
        #'close_ma10_ratio', 'volume_ma10_ratio',
        #'close_ma20_ratio', 'volume_ma20_ratio',
        #'close_ma60_ratio', 'volume_ma60_ratio',
        #'close_ma120_ratio', 'volume_ma120_ratio'
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
