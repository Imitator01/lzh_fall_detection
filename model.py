from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout,Conv2D,MaxPooling2D,Flatten
from keras import layers
from keras import metrics
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from numpy import array
from tensorflow.keras.models import Sequential

def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * np.exp(0.1 * (10 - epoch))


def plotting_training(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    # Plot training & validation loss values
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label="Training loss")
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('epoch', fontsize=10)
    plt.ylabel('loss', fontsize=10)
    plt.ylim(0.0, 0.5)
    plt.legend()
    plt.show()

# train model using CNN
def train_cnn_model(data, label):
    x_train = data
    y_train = np_utils.to_categorical(label)  # one hot 编码
    x_train = array(x_train)
    x_train = x_train.reshape((len(x_train), 2, int(len(x_train[0])/2), 1))###18个点坐标

    y_train = array(y_train)

    cnn_model = Sequential()
    cnn_model.add(Conv2D(64,
                  kernel_size=3,
                  activation='relu',
                  input_shape=(2,18,1),  ###18个点坐标
                  padding='same'))
    cnn_model.add(layers.BatchNormalization(1))
    cnn_model.add(Conv2D(64,
                  kernel_size=3,
                  activation='relu',
                  padding='same'))
    cnn_model.add(layers.BatchNormalization(1))
    cnn_model.add(MaxPooling2D(2,2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(512, activation = 'relu'))
    cnn_model.add(Dense(2, activation='sigmoid')) ###与类别一致

    cnn_model.compile(optimizer='Adam',
                loss='binary_crossentropy',
                metrics=['acc',
                        metrics.AUC(),
                        metrics.FalseNegatives(),
                        metrics.Recall(),
                        metrics.Precision(),
                        metrics.FalseNegatives(),
                        metrics.TrueNegatives(),
                        metrics.FalsePositives(),
                        metrics.TruePositives()])
    cnn_history = cnn_model.fit(x_train, y_train,
                      epochs=100,
                      batch_size=16,
                      validation_split=0.2,
                      callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5),
                      callbacks.LearningRateScheduler(scheduler)])

    print("finish training cnn model")
    return cnn_model, cnn_history

# train model using lstm
def train_lstm_model(data, label):
    x_train = array(data)
    x_train = x_train.reshape((len(x_train), 1, len(x_train[0])))
    print("x_train.shape", x_train.shape)
    print(x_train[0])

    y_train = array(label)
    print("y_train.shape", y_train.shape)

    lstm_model = Sequential()
    lstm_model.add(LSTM(16,
                input_shape=(1, 36),
                return_sequences=True))
    lstm_model.add(LSTM(16, ))
    lstm_model.add(layers.Dense(2, activation='sigmoid'))
    lstm_model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['acc',
                        metrics.AUC(),
                        metrics.FalseNegatives(),
                        metrics.Recall(),
                        metrics.Precision(),
                        metrics.FalseNegatives(),
                        metrics.TrueNegatives(),
                        metrics.FalsePositives(),
                        metrics.TruePositives()])
    lstm_history = lstm_model.fit(x_train, y_train,
                      epochs=100,
                      batch_size=16,
                      validation_split=0.2,
                      callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5),
                      callbacks.LearningRateScheduler(scheduler)])
    print("finish training lstm model")
    return lstm_model, lstm_history