# -*- coding: utf-8 -*-

import numpy as np
from scipy import misc
from PIL import Image
import glob
import matplotlib.pyplot as plt
import scipy.misc
from matplotlib.pyplot import imshow
from IPython.display import SVG
import cv2
import seaborn as sn
import pandas as pd
import pickle
from keras import layers
from keras.layers import Flatten, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Sequential, Model, load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import layer_utils, np_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras import losses
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

from keras.datasets import cifar100

# Red neuronal basica
# https://enmilocalfunciona.io/deep-learning-basico-con-keras-parte-2-convolutional-nets/

def create_simple_nn():
    model = Sequential()
    # La instrucción Flatten convierte los elementos de la matriz de imagenes de entrada en un array plano.
    model.add(Flatten(input_shape=(32, 32, 3), name="Input_layer"))
    # la instrucción Dense, añadimos una capa oculta (hidden layer) de la red neuronal, 1000 neuronas, 500 neuronas
    model.add(Dense(1000, activation='relu', name="Hidden_layer_1"))
    model.add(Dense(500, activation='relu', name="Hidden_layer_2"))
    # Capa de salida
    model.add(Dense(100, activation='softmax', name="Output_layer"))

    return model

# -------------------------------------------------------------------

# Descargar cifar100
(x_train_original, y_train_original), (x_test_original, y_test_original) = cifar100.load_data(label_mode='fine')

# convertir este array en su versión one-hot-encoding, vector 0 y 1 especificando la clase con la posicion del 1
y_train = np_utils.to_categorical(y_train_original, 100)
y_test = np_utils.to_categorical(y_test_original, 100)

print(y_train)
print(x_train_original[3])

# Mostrar imagen de ejemplo
imgplot = plt.imshow(x_train_original[3])
plt.show()

# normalizar las imágenes. Esto es, dividiremos cada elemento de x_train_original por el numero de píxeles, es decir, 255. Con esto obtenemos que el array comprenderá valores de entre 0 y 1
x_train = x_train_original/255
x_test = x_test_original/255

# especificar a Keras dónde se encuentran los canales. En un array de imagenes, pueden venir como ultimo indice o como el primero. Esto se conoce como canales primero (channels first) o canales al final (channels last). En nuestro caso, vamos a definirlos al final.
K.set_image_data_format('channels_last')

# especificar es la fase del experimento. En este caso, la fase será de entrenamiento.
K.set_learning_phase(1)

snn_model = create_simple_nn()
# funcion de optimizacion: stochactic gradient descendent
# funcion de perdida: categorical cross entropy
# accuracy y mean squared error para metricas
snn_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc', 'mse'])

# Resumen del modelo creado
snn_model.summary()

# Ejecutar entrenamiento
snn = snn_model.fit(x=x_train, y=y_train, batch_size=32, epochs=10, verbose=1, validation_data=(x_test, y_test), shuffle=True)

# Evaluar con otro dataset
evaluation = snn_model.evaluate(x=x_test, y=y_test, batch_size=32, verbose=1)

plt.figure(0)
# train accuracy
plt.plot(snn.history['acc'], 'r')
# validation accuracy
plt.plot(snn.history['val_acc'], 'g')
plt.xticks(np.arange(0, 11, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train', 'validation'])

plt.figure(1)
# training loss
plt.plot(snn.history['loss'], 'r')
# validation loss
plt.plot(snn.history['val_loss'], 'g')
plt.xticks(np.arange(0, 11, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train', 'validation'])

plt.show()

# Matriz de confusion, prediccion sobre el dataset de validacion
snn_pred = snn_model.predict(x_test, batch_size=32, verbose=1)
snn_predicted = np.argmax(snn_pred, axis=1)

#Creamos la matriz de confusión
snn_cm = confusion_matrix(np.argmax(y_test, axis=1), snn_predicted)

# Visualizamos la matriz de confusión, valores de precision, recall y f1-score
snn_df_cm = pd.DataFrame(snn_cm, range(100), range(100))
plt.figure(figsize = (20,14))
sn.set(font_scale=1.4) #for label size
sn.heatmap(snn_df_cm, annot=True, annot_kws={"size": 12}) # font size
plt.show()

# Mostrar metricas para cada clase
snn_report = classification_report(np.argmax(y_test, axis=1), snn_predicted)
print(snn_report)

# Ver resultados
imgplot = plt.imshow(x_train_original[0])
plt.show()
print('class for image 1: ' + str(np.argmax(y_test[0])))
print('predicted:         ' + str(snn_predicted[0]))

# Ver resultados
imgplot = plt.imshow(x_train_original[3])
plt.show()
print('class for image 3: ' + str(np.argmax(y_test[3])))
print('predicted:         ' + str(snn_predicted[3]))