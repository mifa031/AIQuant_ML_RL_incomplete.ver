import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Dropout#, BatchNormalization, LSTM
from keras.optimizers import sgd
from keras import backend as K

class ValueNetwork:
    def __init__(self, input_dim=0, lr=0.01):
        self.input_dim = input_dim
        self.lr = lr

    def make_model(self):
        # 심층 신경망
        self.model = Sequential()
        self.model.add(Dense(8192, input_dim=self.input_dim, activation='elu'))
        self.model.add(Dropout(0.5))
        
        self.model.add(Dense(4096, activation='elu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='elu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2048, activation='elu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2048, activation='elu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='elu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='elu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='elu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='elu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='elu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='elu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='elu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='elu'))
        self.model.add(Dropout(0.5))
        
        self.model.add(Dense(1, activation='linear'))

        self.prob = None

        self.model.summary()
        return self.model

    def reset(self):
        self.prob = None

    def predict(self, sample):
        self.prob = self.model.predict(np.array(sample).reshape((1,self.input_dim)))[0]
        return self.prob

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)
