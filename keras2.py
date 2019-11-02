# -*- coding: utf-8 -*-
import pickle

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report


# ConvNet
# https://enmilocalfunciona.io/deep-learning-basico-con-keras-parte-2-convolutional-nets/


def create_simple_cnn():
    model = Sequential()
    # Capa convolucional de entrada
    # 32 filtros
    # filtro tamaño 3x3
    # activacion relu
    # total_params = (filter_height * filter_width * input_image_channels + 1) * number_of_filters
    # Param number = 896
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu'))
    # Capa convolucional
    # 64 filtros
    # filtro tamaño 3x3
    # activacion relu
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # Capa max pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    # Dropout consists in randomly setting a fraction `rate` of input units to 0 at each update during training time,
    # which helps prevent overfitting.
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='softmax'))

    return model


# -------------------------------------------------------------------


from keras.datasets import cifar100

(x_train_original, y_train_original), (x_test_original, y_test_original) = cifar100.load_data(label_mode='fine')

# One hot encoding
y_train = np_utils.to_categorical(y_train_original, 100)
y_test = np_utils.to_categorical(y_test_original, 100)

imgplot = plt.imshow(x_train_original[3])
plt.show()

# Normalizar imagenes
x_train = x_train_original / 255
x_test = x_test_original / 255

# Canales de imagenes
K.set_image_data_format('channels_last')

# Fase del experimento
K.set_learning_phase(1)

# Compilar el modelo
scnn_model = create_simple_cnn()

# funcion de optimizacion: stochactic gradient descendent
# funcion de perdida: categorical cross entropy
# accuracy y mean squared error para metricas
scnn_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc', 'mse'])

scnn_model.summary()

# Entrenar la red
# 32 batches o bloques, 10 vueltas completas o epochs
scnn = scnn_model.fit(x=x_train, y=y_train, batch_size=32, epochs=10, verbose=1, validation_data=(x_test, y_test),
                      shuffle=True)

# Evaluar con otro dataset
cnn_evaluation = scnn_model.evaluate(x=x_test, y=y_test, batch_size=32, verbose=1)

plt.figure(0)
plt.plot(scnn.history['acc'], 'r')
plt.plot(scnn.history['val_acc'], 'g')
plt.xticks(np.arange(0, 11, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train', 'validation'])

plt.figure(1)
plt.plot(scnn.history['loss'], 'r')
plt.plot(scnn.history['val_loss'], 'g')
plt.xticks(np.arange(0, 11, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train', 'validation'])

plt.show()

# Matriz de confusion
scnn_pred = scnn_model.predict(x_test, batch_size=32, verbose=1)
scnn_predicted = np.argmax(scnn_pred, axis=1)

# Creamos la matriz de confusión
scnn_cm = confusion_matrix(np.argmax(y_test, axis=1), scnn_predicted)

# Visualiamos la matriz de confusión
scnn_df_cm = pd.DataFrame(scnn_cm, range(100), range(100))
plt.figure(figsize=(20, 14))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(scnn_df_cm, annot=True, annot_kws={"size": 12})  # font size
plt.show()

scnn_report = classification_report(np.argmax(y_test, axis=1), scnn_predicted)
print(scnn_report)


# Guardar historico en txt
with open('scnn_history.txt', 'wb') as file_pi:
  pickle.dump(scnn.history, file_pi)