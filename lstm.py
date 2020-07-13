import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from make_data import *


class HyperParameter(object):
    def __init__(self):
        self.rates = [0, 0.25, 0.5, 0.75, 1]

        self.input_size = 100
        self.output_size = len(self.rates)

        self.epoch = 50
        self.learning_rate = 0.01
        self.lstm_size = 100

        self.data_path = ''


def build_model(params: HyperParameter):
    m = keras.Sequential()
    m.add(layers.LSTM(params.lstm_size))
    m.add(layers.Dense(params.output_size, activation='softmax'))

    m.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(params.learning_rate),
              metrics=['mse', 'accuracy'])

    return m


def plot_accuracy(h):
    plt.plot(h.history['accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def plot_loss(h):
    plt.plot(h.history['loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    param = HyperParameter()
    model = build_model(param)

    x = []
    y = []
    for rate in param.rates:
        for i in range(5):
            x.append(make_data_set(rate, period=param.input_size/5, cycle=5))
            y.append(param.rates.index(rate))

    x = np.array(x, dtype=np.float32).reshape((-1, param.input_size, 1))
    y = np.eye(len(param.rates), dtype=np.float32)[y]

    history = model.fit(x, y, batch_size=1, epochs=param.epoch)
    pred = model.predict(
        np.array(
            make_data_set(0.5, period=param.input_size/5, cycle=5), dtype=np.float32
        ).reshape((-1, param.input_size, 1))
    )
    print(pred)

    plot_accuracy(history)
    plot_loss(history)
