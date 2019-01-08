import numpy as np
import random

class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 0  # 코인 보유 비율, #기준 포트폴리오 가치 비율, 시작 포트폴리오 가치 비율

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0  # 거래 수수료 미고려 (일반적으로 0.015%)
    TRADING_TAX = 0  # 거래세 미고려 (실제 0.3%)

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩
    ACTIONS = [ACTION_BUY, ACTION_SELL]  # 인공 신경망에서 확률을 구할 행동들
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(
        self, environment):
        # Environment 객체
        self.environment = environment  # 현재 주식 가격을 가져오기 위해 환경 참조

        # Agent 클래스의 속성
        self.initial_balance = 0  # 초기 자본금
        self.balance = 0  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        self.portfolio_value = 0  # balance + num_stocks * {현재 주식 가격}
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상
        self.confidence = 0
        self.rand_action = -1

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        #self.ratio_hold = float(self.num_stocks) / (self.portfolio_value / float(self.environment.get_price()))
        #self.ratio_portfolio_value = self.portfolio_value / self.base_portfolio_value
        #self.global_ratio_portfolio_value = self.portfolio_value / self.initial_balance
        global_profitloss = ((self.portfolio_value - self.initial_balance) / self.initial_balance)
        return (
            #self.ratio_hold,
            #self.ratio_portfolio_value,
            #self.global_ratio_portfolio_value
            [global_profitloss]
        )

    def decide_action(self, policy_network_obj, state):
        policy = policy_network_obj.predict(state).flatten()
        return policy

    def act(self, action, epsilon):
        if np.random.rand() < epsilon:
            exploration = True
            self.confidence = random.random()
            action = np.random.randint(self.NUM_ACTIONS)  # 무작위로 행동 결정
            self.rand_action = action
        else:
            exploration = False
            action_idx = np.random.choice(self.NUM_ACTIONS, 1, p=action)[0]
            self.confidence = action[action_idx]
            action = action_idx

        # 환경에서 현재 가격 얻기
        self.curr_price = self.environment.get_price()
        curr_price = self.curr_price
        
        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            trading_unit = self.balance / (curr_price * (1 + self.TRADING_CHARGE))
            balance = self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            if balance < 0:
                trading_unit = 0
            
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            self.balance -= invest_amount  # 보유 현금을 갱신
            self.num_stocks += trading_unit  # 보유 주식 수를 갱신
            self.num_buy += 1  # 매수 횟수 증가

        # 매도
        elif action == Agent.ACTION_SELL:
            # 남은 코인 모두 매도
            trading_unit = self.num_stocks
            invest_amount = curr_price * (1 - (self.TRADING_CHARGE)) * trading_unit

            self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
            self.balance += invest_amount  # 보유 현금을 갱신
            self.num_sell += 1  # 매도 횟수 증가

        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

        self.next_price = self.environment.get_next_price()
        next_price = self.next_price
        if next_price is not None:
            # 포트폴리오 가치 갱신
            self.portfolio_value = self.balance + next_price * self.num_stocks
            profitloss = ((self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)
            #global_profitloss = ((self.portfolio_value - self.initial_balance) / self.initial_balance)

            #즉시 보상 판단 #다음날을 보니 이익,손해가 이렇다
            self.immediate_reward = profitloss

        self.base_portfolio_value = self.portfolio_value

        return self.immediate_reward, exploration, action_idx
