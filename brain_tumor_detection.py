# -*- coding: utf-8 -*-
"""Brain_Tumor_detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10-y8MOptXcEzZReKVY-F-4gnh_fn8Ur1
"""

import numpy as np # linear algebra
import pandas as pd 
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import  normalize
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator
import glob
import shutil
import random
import matplotlib.pyplot as plt
import cv2
np.random.seed(42)

from google.colab import drive
drive.mount('/content/drive')

vgg_model=tf.keras.applications.mobilenet.MobileNet()

train_path='/content/drive/MyDrive/Brain_tumor/train'
test_path='/content/drive/MyDrive/Brain_tumor/test'
valid_path='/content/drive/MyDrive/Brain_tumor/valid'
train_batch= ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
  .flow_from_directory(directory=train_path, target_size=(64,64), classes=['no','yes'],batch_size=10)
test_batch= ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
  .flow_from_directory(directory=test_path, target_size=(64,64), classes=['no','yes'],batch_size=10, shuffle=False)
valid_batch= ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
  .flow_from_directory(directory=valid_path, target_size=(64,64), classes=['no','yes'],batch_size=10)

assert train_batch.n ==2400
assert test_batch.n ==200
assert valid_batch.n==400
imgs,lables=next(train_batch)
def plotImages(images_arr):
  fig, axes= plt.subplots(1,10,figsize=(20,20))
  axes= axes.flatten()
  for img, ax in zip(images_arr,axes):
    ax.imshow(img)
    ax.axis('off')
  plt.tight_layout()
  plt.show()
plotImages(imgs)
print(lables)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(64,64,3)),
    MaxPooling2D(pool_size=(2,2), strides=1),
    Conv2D(filters=64, kernel_size=(3,3), activation='relu',padding='same'),
    MaxPooling2D(pool_size=(2,2),strides=1),
    Flatten(),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])  
model.summary()

decay_learning= tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9,
)
adam=tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'],)

model.fit(x=train_batch, validation_data=valid_batch, epochs=25, verbose=2)

history=model.history.history
for i in history.keys():
  print(i)

def plot_metrics(history):
    
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_acc = history['accuracy']
    val_acc = history['val_accuracy']
    
    # Loss
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    
    # Accuracy
    plt.figure()
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

plot_metrics(history)

predictions=model.predict(test_batch)

np.round(predictions)

from sklearn.metrics import confusion_matrix, classification_report
cm=confusion_matrix(y_pred=np.argmax(predictions, axis=-1), y_true=test_batch.classes)
cm

from sklearn.metrics import accuracy_score
aoc=accuracy_score(y_pred=np.argmax(predictions, axis=-1),y_true=test_batch.classes)
aoc

print(classification_report(test_batch.classes,np.argmax(predictions, axis=-1),labels=[0,1],digits=3))

model.save('my_model.h5')

model.summary()