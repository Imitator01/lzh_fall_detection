from model import train_lstm_model, train_cnn_model, plotting_training
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import csv
from numpy import array
import pandas as pd
from lagrange import lagrange

def load_data(path):
    # load data
    raw_data = pd.read_csv(path, header=0)
    dataset = raw_data.values

    data = dataset[0:2462, 0:36].astype(float)
    label = dataset[0:2462, 36]

    label = array(label)
    label = np_utils.to_categorical(label)# one hot 编码

    return data,label

def training(data,label,save_model):
    # train lstm model
    lstm_model, lstm_history = train_lstm_model(data, label)
    lstm_model.summary()
    if save_model == True:
        lstm_model.save("./model_lstm.h5")
    plotting_training(lstm_history)

def test(x_test,y_test,lstm_model):

    x_test = array(x_test)
    x_test = x_test.reshape((len(x_test), 3, int(len(x_test[0])/3), 1))

    y_test = array(y_test)

    test_score = lstm_model.evaluate(x_test, y_test)
    print(test_score)

def inter(x,y):

# x = [532.109, 543.9259999999999, 517.06, 546.529, 515.752, 541.934, 511.211, 542.6080000000001, 530.82, 528.201]
# y = [122.28399999999999, 117.014, 128.857, 117.016, 132.083, 118.335, 132.752, 117.662, 122.955, 123.579]
# # plt.plot(x,y)
# # plt.show()


    x_test = list(np.linspace(x[0], x[-1], 50))
    y_predict = [lagrange(x, y, len(x), x_i) for x_i in x_test]

    return y_predict

if __name__ == '__main__':

    datapath = './train_data.csv'
    data, label = load_data(datapath)

      ## train
    training(data,label,save_model=False)


    x_test = []
    y_test = []
    for i in range(10):
        x_test.append(data[i][0])
        y_test.append(data[i][1])


