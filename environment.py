class Environment:

    PRICE_IDX = 4  # 종가의 위치

    def __init__(self, chart_data=None):
        self.chart_data = chart_data
        self.observation = None
        self.idx = -1

    def reset(self, num_past_input):
        self.observation = None
        self.idx = -1 + num_past_input

    def observe(self):
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None
    
    def get_next_price(self):
        try:
            return self.chart_data.iloc[self.idx+1][self.PRICE_IDX]
        except IndexError as e :
            return None

    def set_chart_data(self, chart_data):
        self.chart_data = chart_data
