import numpy as np
import pandas as pd
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from os import listdir
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt
from skimage import io
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D




#loading up model
target_width = 200
target_height = 200

def reshape_image(filePath):
  desired_size = 200
  im = cv2.imread(filePath)
  old_size = im.shape[:2] # old_size is in (height, width) format
  ratio = float(desired_size)/max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])
  # new_size should be in (width, height) format
  im = cv2.resize(im, (new_size[1], new_size[0]))
  delta_w = desired_size - new_size[1]
  delta_h = desired_size - new_size[0]
  top, bottom = delta_h//2, delta_h-(delta_h//2)
  left, right = delta_w//2, delta_w-(delta_w//2)
  color = [0, 0, 0]
  new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
  gray= cv2.cvtColor(new_im,cv2.COLOR_BGR2GRAY)
  return gray


all_images = []
for image_path in listdir('./2013-02-22/Empty/'):
  result = reshape_image('./2013-02-22/Empty/'+image_path)
  #print(result)
  #print(type(result))
  #print(result.shape)
  #plt.imshow(result,cmap='gray')
  #plt.show()
  all_images.append(result)
y_train_empty = np.array([0]*len(all_images))
temp_size = len(all_images)
for image_path in listdir('./2013-02-22/Occupied/'):
  result = reshape_image('./2013-02-22/Occupied/'+image_path)
  #print(type(result))
  #print(result.shape)
  #plt.imshow(result,cmap='gray')
  #plt.show()
  all_images.append(result)
y_train_occupied = np.array([1]* (len(all_images)-temp_size))
print('length of all_images', len(all_images))
X = np.array(all_images)
print(type(X))
print(X.shape)
y = np.concatenate((y_train_empty, y_train_occupied), axis=None)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# setting up CNN
cnn_layers = [Conv2D(32, (5,5),activation='relu',padding='same',input_shape=(target_width,target_height,1))]
cnn_layers.append(MaxPool2D())
cnn_layers.append(Conv2D(64, (5,5),activation='relu',padding='same'))
cnn_layers.append(MaxPool2D())
cnn_layers.append(Conv2D(128, (5,5),activation='relu',padding='same'))
cnn_layers.append(MaxPool2D())
cnn_layers.append(Conv2D(128, (5,5),activation='relu',padding='same'))
cnn_layers.append(MaxPool2D())
cnn_layers.append(Flatten())
cnn_layers.append(Dense(100,activation='relu'))
cnn_layers.append(Dense(100,activation='relu'))
cnn_layers.append(Dense(1,activation='softmax'))
cnn_model = Sequential(cnn_layers)

#training CNN
cnn_model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train.reshape(-1,200,200,1), y_train, epochs=1)


#evaluate CNN
cnn_scores = cnn_model.evaluate(X_test.reshape(-1, 200, 200 ,1), y_test)
print('accuracy:',cnn_scores[1])



#empty = np.ndarray([tf.image.resize_image_with_crop_or_pad(cv.imread(fileName),target_height,target_width) for fileName in listdir('./2013-02-22/Empty/')])
#print(empty)