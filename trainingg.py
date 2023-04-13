import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2

from keras.engine.functional import Functional
import tensorflow as tf
from PIL import Image
import os
os.chdir('C:/Users/INDIAN/Desktop/softwarelab')
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

from sklearn.metrics import accuracy_score
print('hello')

data = []
labels = []
# We have 43 Classes
classes = 43
cur_path = os.getcwd()
print('hello')
for i in range(classes):
    path = os.path.join(cur_path,'Train',str(i))
    images = os.listdir(path)
    for a in images:
        try:
            print(i)
            image = Image.open(path + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(e)


data = np.array(data)
labels = np.array(labels)



print('hello')


# os.mkdir('training')

np.save('./training/data',data)
np.save('./training/target',labels)


data=np.load('./training/data.npy')
labels=np.load('./training/target.npy')



print(data.shape, labels.shape)
# trainging 20  and 80 testing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# categorical convert into array of row and column
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
model = Sequential()
# relu is rectified linear unit  activation function
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
# flatten is used for converting metrices in array
model.add(Dense(256, activation='relu'))

model.add(Dropout(rate=0.5))
# We have 43 classes that's why we have defined 43 in the dense
model.add(Dense(43, activation='softmax'))
# softmax mathematical function that a vector of numbers into a
#  vector of probabitlites of each value propotional to relative scale
# cate- used as a loss function or to more output label
# adam adaptive optimization algorithm created for deep neural network 
#  


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 20
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))





model.save("./training/traffic_classifier.h5")

print('Model trained successfully') 


