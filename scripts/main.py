import numpy as np
import pandas as pd
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from os import listdir

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import scipy
import cv2 as cv
from matplotlib import pyplot as plt



img = cv.imread('venice.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create() #docs: https://docs.opencv.org/3.4/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
kp, des = sift.detectAndCompute(gray,None)
print(len(kp))
print(len(des))
img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv.imwrite('sift_keypoints.jpg',img) # writes to a file
plt.imshow(gray)
plt.show()




'''
#n,w,h = x_train.shape
target_width = 100
target_height = 100

tf.image.resize_image_with_crop_or_pad(image,target_height,target_width)


cnn_layers = [Conv2D(32, (3,3),activation='relu',padding='same',input_shape=(w,h,1))]
cnn_layers.append(MaxPool2D())
cnn_layers.append(Conv2D(64, (3,3),activation='relu',padding='same'))
cnn_layers.append(MaxPool2D())
cnn_layers.append(Conv2D(128, (3,3),activation='relu',padding='same'))
cnn_layers.append(MaxPool2D())
cnn_layers.append(Conv2D(128, (3,3),activation='relu',padding='same'))
cnn_layers.append(MaxPool2D())
cnn_layers.append(Flatten())
cnn_layers.append(Dense(200,activation='relu'))
cnn_layers.append(Dense(200,activation='relu'))
cnn_layers.append(Dense(20,activation='softmax'))
cnn_model = Sequential(cnn_layers)

cnn_model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(x_train.reshape(-1, 28, 28 ,1), y_train, epochs=1)
'''