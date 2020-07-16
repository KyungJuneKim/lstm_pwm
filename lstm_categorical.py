import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense

from make_data import make_data_set
from util import split_data, plot_model


class HyperParameter:
    def __init__(self):
        self.rates = [0, 0.25, 0.5, 0.75, 1]
        self.period = 20
        self.cycle = 5

        self.input_size = self.period * self.cycle
        self.output_size = len(self.rates)

        self.epoch = 5
        self.learning_rate = 0.01
        self.lstm_size = 100

        self.data_path = ''


def build_model(params: HyperParameter):
    m = keras.Sequential()
    m.add(LSTM(params.lstm_size))
    m.add(Dense(params.output_size, activation=keras.activations.softmax))

    m.compile(loss=keras.losses.CategoricalCrossentropy(),
              optimizer=keras.optimizers.RMSprop(params.learning_rate),
              metrics=['mse', 'accuracy'])

    return m


if __name__ == '__main__':
    param = HyperParameter()
    model = build_model(param)

    x_train = []
    y_train = []
    x_val = []
    y_val = []

    for rate in param.rates:
        x = []
        y = []
        for i in range(200):
            x.append(make_data_set(rate, period=param.period, cycle=param.cycle))
            y.append(param.rates.index(rate))

        (x_train_tmp, y_train_tmp), (x_val_tmp, y_val_tmp) = split_data(x, y, ratio=[0.8])
        x_train += x_train_tmp
        y_train += y_train_tmp
        x_val += x_val_tmp
        y_val += y_val_tmp

    x_train = np.array(x_train, dtype=np.float32).reshape((-1, param.input_size, 1))    # reshape
    y_train = np.eye(len(param.rates), dtype=np.float32)[y_train]                       # to one-hot
    x_val = np.array(x_val, dtype=np.float32).reshape((-1, param.input_size, 1))
    y_val = np.eye(len(param.rates), dtype=np.float32)[y_val]

    history = model.fit(x_train, y_train, batch_size=1, epochs=param.epoch, validation_data=(x_val, y_val))

    pred = model.predict(
        np.array(
            make_data_set(0.75, period=param.period, cycle=param.cycle), dtype=np.float32
        ).reshape((-1, param.input_size, 1))
    )
    print(pred)

    plot_model(history, validation=True, keys=['loss', 'mse', 'accuracy'])
