import numpy as np
from random import random
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense

from make_data import make_data_set
from util import split_data, plot_model


class HyperParameter:
    def __init__(self):
        self.period = 20
        self.cycle = 5

        self.input_size = self.period * self.cycle
        self.output_size = 1

        self.epoch = 5
        self.learning_rate = 0.01
        self.lstm_size = 100

        self.data_path = ''


def build_model(params: HyperParameter):
    m = keras.Sequential()
    m.add(LSTM(params.lstm_size))
    m.add(Dense(params.output_size, activation=keras.activations.sigmoid))

    m.compile(loss='mse',
              optimizer=keras.optimizers.Adam(params.learning_rate),
              metrics=['mae'])

    return m


if __name__ == '__main__':
    param = HyperParameter()
    model = build_model(param)

    x = []
    y = []
    for i in range(2000):
        ratio = random()
        x.append(make_data_set(ratio, period=param.period, cycle=param.cycle))
        y.append(ratio)

    (x_train, y_train), (x_val, y_val) = split_data(x, y, ratio=[0.9])

    x_train = np.array(x_train, dtype=np.float32).reshape((-1, param.input_size, 1))
    y_train = np.array(y_train, dtype=np.float32)
    x_val = np.array(x_val, dtype=np.float32).reshape((-1, param.input_size, 1))
    y_val = np.array(y_val, dtype=np.float32)

    history = model.fit(x_train, y_train, batch_size=1, epochs=param.epoch, validation_data=(x_val, y_val))

    pred = model.predict(
        np.array(
            make_data_set(0.75, period=param.period, cycle=param.cycle), dtype=np.float32
        ).reshape((-1, param.input_size, 1))
    )
    print(pred)

    plot_model(history, validation=True, keys=['loss', 'mae'])
