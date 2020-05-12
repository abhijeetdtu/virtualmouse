%load_ext autoreload
autoreload 2

import numpy as np

import pickle

from virtualmouse.constants import Constants ,  ModelConsts
from virtualmouse.utils import *

import pathlib
import os

try:
    __file__
except:
    __file__ = "./virtualmouse/process.py"

path = os.path.abspath(os.path.join(os.path.dirname(pathlib.Path(__file__).resolve().absolute()) , ".."))

allData = []
for f in os.listdir(path):
    if f.find(Constants.IMAGE_DUMP_FILE) > -1:
        p = os.path.join(path , f)
        data = pickle.load(open(p, "rb"))
        allData.extend(data)

#len(allData)
images = []
labels = []
for image,c in allData:
    images.append(image)
    labels.append(c[0])

#images[0].shape
# len(images)
#
# len(labels)
#
# images = np.array(images)
# labels = np.array(labels)
#
# images.shape

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler((0,1))

import cv2

images = np.array([cv2.resize(img,None,fx=0.2 , fy=0.2) for img in images])

labels = scaler.fit_transform(labels)
X_train,X_test , y_train,y_test= train_test_split(images , labels)

X_train[0].shape
y_train[0]
#t = X_train.reshape(len(X_train) , 1 ,
#np.prod(X_train[0].shape)

#y_train[0]

import keras
from keras.layers import MaxPooling2D,Dropout,Flatten,Dense

#input_shape = (1, X_train[0].shape[0] , X_train[0].shape[1],X_train[0].shape[2])

def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64,kernel_size=(75,75),input_shape=X_train[0].shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(filters=32,kernel_size=(25,25)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(filters=16,kernel_size=(5,5)))
    #model.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model = CompileModel(model)
    return model

model = build_model()
model.summary()
model.fit(X_train,y_train, epochs=10)

from sklearn.metrics import mean_squared_error

mean_squared_error(model.predict(X_test) , y_test)


from virtualmouse.utils import *

SaveModel(model)

model = LoadModel()
model.predict(X_test)
#model.save(ModelConsts.ModelConsts.MODEL_JS_LOCATION)
