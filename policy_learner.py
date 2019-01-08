import os
import locale
import logging
import pylab
import numpy as np
import settings
from environment import Environment
from agent import Agent
from policy_network import PolicyNetwork
from value_network import ValueNetwork
from keras import backend as K
from keras.optimizers import Adam, RMSprop, Nadam
import time
import random
from keras.models import load_model

logger = logging.getLogger(__name__)
locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')

class PolicyLearner:
    def __init__(self, stock_code, chart_data, training_data=None, policy_model_path=None, value_model_path=None,
                 lr=0.001, discount_factor=0.5, start_epsilon=0, num_past_input=0,
                 load_weight_and_learn=False):
        self.stock_code = stock_code  # 종목코드
        self.chart_data = chart_data
        self.environment = Environment(chart_data)  # 환경 객체
        # 에이전트 객체
        self.agent = Agent(self.environment)
        self.training_data = training_data  # 학습 데이터
        self.training_data_idx = -1
        self.state = None
        self.action_size = self.agent.NUM_ACTIONS
        self.discount_factor = discount_factor
        self.start_epsilon = start_epsilon
        self.num_past_input = num_past_input
        self.load_weight_and_learn = load_weight_and_learn
        
        # 정책/가치 신경망; 입력 크기 = 학습 데이터의 크기 #+ 에이전트 상태 크기
        self.num_features = self.training_data.shape[1] * (1+num_past_input) #+ self.agent.STATE_DIM
        self.policy_network_obj = PolicyNetwork(input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=lr)
        if load_weight_and_learn is True:
            self.policy_network_obj.model = load_model(policy_model_path)
            self.policy_network = self.policy_network_obj.model
        else:
            self.policy_network = self.policy_network_obj.make_model()
        self.value_network_obj = ValueNetwork(input_dim=self.num_features, lr=lr)
        if load_weight_and_learn is True:
            self.value_network_obj.model = load_model(value_model_path)
            self.value_network = self.value_network_obj.model
        else:
            self.value_network = self.value_network_obj.make_model()
        self.policy_updater = self.policy_optimizer()
        self.value_updater = self.value_optimizer()

    # 정책신경망을 업데이트하는 함수
    def policy_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantage = K.placeholder(shape=[None, ])

        action_prob = K.sum(action * self.policy_network.output, axis=1)
        cross_entropy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entropy)

        optimizer = Nadam(lr=self.policy_network_obj.lr)
        updates = optimizer.get_updates(self.policy_network.trainable_weights, [], loss)
        train = K.function([self.policy_network.input, action, advantage], [],
                           updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def value_optimizer(self):
        target = K.placeholder(shape=[None, ])

        loss = K.mean(K.square(target - self.value_network.output))

        optimizer = Nadam(lr=self.value_network_obj.lr)
        updates = optimizer.get_updates(self.value_network.trainable_weights, [], loss)
        train = K.function([self.value_network.input, target], [], updates=updates)

        return train


    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, action_idx, reward, next_state, done):
        value = self.value_network_obj.predict(state)[0]
        #next_value = self.value_network_obj.predict(next_state)[0]
        next_value = 0
        
        act = np.zeros([1,self.action_size])
        #action_idx = np.random.choice(self.action_size, 1, p=action)[0]
        act[0][action_idx] = 1

        # 벨만 기대 방정식를 이용한 어드벤티지와 업데이트 타깃
        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = (reward + self.discount_factor * next_value) - value
            target = [reward + self.discount_factor * next_value]
            
        state_list = [state]
        advantage_list = [advantage]
        self.policy_updater([state_list, act, advantage_list])
        self.value_updater([state_list, target])
    
    def reset(self):
        self.sample = None
        self.training_data_idx = -1 + self.num_past_input

    def trade(self, model_path=None, balance=2000000):
        if model_path is None:
            return
        self.policy_network_obj.model = load_model(model_path)
        self.fit(balance=balance, num_epoches=1, learning=False)

    def fit(self, num_epoches=1000, balance=10000000,learning=True):
        logger.info("LR: {lr}, DF: {discount_factor}, "
                    "TU: all-in, "
                    "DRT: only-use immediate reward".format(
            lr=self.policy_network_obj.lr,
            discount_factor=self.discount_factor
        ))
        
        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0
        portfolio_repeat_cnt = 0
        exploration = False
        episode_results, episodes = [], []
        pylab.clf()
        # 학습 반복
        for epoch in range(num_epoches):
            start_time = time.time()
            # 에포크 관련 정보 초기화
            previous_portfolio_value = self.agent.portfolio_value
            loss = 0.
            itr_cnt = 0
            win_cnt = 0
            exploration_cnt = 0
            batch_size = 0
            pos_learning_cnt = 0
            neg_learning_cnt = 0
            
            # 환경, 에이전트, 정책 신경망 초기화
            self.environment.reset(self.num_past_input)
            self.agent.reset()
            self.policy_network_obj.reset()
            self.value_network_obj.reset()
            self.reset()

            self.environment.observe()
            self.training_data_idx += 1
            self.state = []
            for i in range(self.num_past_input+1):
                self.state.extend(self.training_data.iloc[self.training_data_idx-i].tolist())
            #self.state.extend(self.agent.get_states())
            done = False         
            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = self.start_epsilon * (1. - float(epoch) / (num_epoches - 1))
            else:
                epsilon = 0
            #학습 시작
            while True:
                # 정책 신경망에 의한 행동 결정
                self.action = self.agent.decide_action(self.policy_network_obj, self.state)
                # 결정한 행동을 수행하고 즉시 보상 획득
                immediate_reward, exploration, action_idx = self.agent.act(self.action, epsilon)
                if exploration:
                    self.action[self.agent.rand_action] = self.agent.confidence
                    for i in range(self.action_size):
                        if i != self.agent.rand_action:
                            self.action[i] = 1-self.agent.confidence

                # 비학습 모드일 경우
                if learning is False:
                    print(self.environment.chart_data.iloc[self.environment.idx]['date'])
                    print(self.action)
    
                # 반복에 대한 정보 갱신
                itr_cnt += 1
                win_cnt += 1 if immediate_reward > 0 else 0
                exploration_cnt += 1 if exploration is True else 0

                # next state data 생성
                state = self.state #현재 상태인 self.state를 state에 저장
                action = self.action
                observation=self.environment.observe()
                if observation is not None:
                    self.training_data_idx += 1
                    self.state = []
                    for i in range(self.num_past_input+1):
                        self.state.extend(self.training_data.iloc[self.training_data_idx-i].tolist())
                    #self.state = self.training_data.iloc[self.training_data_idx].tolist()
                    #self.state.extend(self.agent.get_states())
                    next_state = self.state
                else:
                    break
                # 학습중이고 랜덤탐헝이 아닌 경우
                if learning and (exploration is False):
                    if immediate_reward > 0:
                        pos_learning_cnt += 1
                    else:
                        neg_learning_cnt += 1                 
                    # 정책 신경망 갱신
                    self.train_model(state, action, action_idx, immediate_reward, next_state, done)

            # 에포크 관련 정보 가시화
            print("epoch:",epoch+1," / sequence:",itr_cnt," / portfolio_value:",self.agent.portfolio_value)
            epoch_time = time.time() - start_time
            remain_time = epoch_time * (num_epoches - (epoch+1))
            print("epoch_time: %s second" %(round(epoch_time,2))," / remain_time: %s hour" %(round(remain_time/3600,2)))
            if(epoch_time > 1): #한 epoch 당 1초가 넘을때만 plot을 그린다
                episode_results.append(self.agent.portfolio_value)
                episodes.append(epoch+1)
                pylab.plot(episodes, episode_results, 'b')
                if not os.path.isdir("./save_graph"):
                    os.makedirs("./save_graph")
                pylab.savefig("./save_graph/result.png")
            
            # 학습 관련 정보 갱신
            max_portfolio_value = max(max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1
                
            # 학습을 일찍 끝낼지 여부 결정
            #if previous_portfolio_value == self.agent.portfolio_value:
            #    portfolio_repeat_cnt += 1
            #else:
            #    portfolio_repeat_cnt = 0
            #if portfolio_repeat_cnt == 10:
            #    break

        # 학습 관련 정보 로그 기록
        logger.info("Max PV: %s, \t # Win: %d" % (locale.currency(max_portfolio_value, grouping=True), epoch_win_cnt))

