import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC # This is the SVM classifier
from matplotlib import pyplot as plt
import glob
import pickle
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def reshape_image(filePath):
  desired_size = 250
  im = cv.imread(filePath)
  print(im.shape)
  old_size = im.shape[:2] # old_size is in (height, width) format
  ratio = float(desired_size)/max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])
  # new_size should be in (width, height) format
  im = cv.resize(im, (new_size[1], new_size[0]))
  delta_w = desired_size - new_size[1]
  delta_h = desired_size - new_size[0]
  top, bottom = delta_h//2, delta_h-(delta_h//2)
  left, right = delta_w//2, delta_w-(delta_w//2)
  color = [0, 0, 0]
  new_im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT,value=color)
  gray= cv.cvtColor(new_im,cv.COLOR_BGR2GRAY)
  return gray

def sift(*args, **kwargs):
    try:
        return cv.xfeatures2d.SIFT_create(*args, **kwargs)
    except:
        return cv.SIFT()
def dsift(img, step=5):
    keypoints = [
        cv.KeyPoint(x, y, step)
        for y in range(0, img.shape[0], step)
        for x in range(0, img.shape[1], step)
    ]
    features = sift().compute(img, keypoints)[1]
    features /= features.sum(axis=1).reshape(-1, 1)
    return features


X = []
y = []
#sift = cv.xfeatures2d.SIFT_create() #docs: https://docs.opencv.org/3.4/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
#dense=cv.FeatureDetector_create("Dense")
i = 0;
for filename in glob.glob('../images/empty/*'):
	if(i < 5):
		
		#img = cv.imread(filename)
		#gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
		img = reshape_image(filename)
		#cv.imshow('img',img)
		#cv.waitKey(0)
		des = dsift(img)
		#if(i == 0):
			#plt.imshow(img, cmap='gray')
			#plt.show()
			#print(des)
			
		#kp=dense.detect(gray)
		#kp,des=sift.compute(gray,kp)
		X.append(des)
		y.append(0)
		i = i + 1
		#print(des)
for filename in glob.glob('../images/occupied/*'):
	if(i < 10):
		#img = cv.imread(filename)
		#gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
		img = reshape_image(filename)
		des = dsift(img)	
		#kp=dense.detect(gray)
		#kp,des=sift.compute(gray,kp)
		X.append(des)
		y.append(1)
		i = i + 1
		#print(des)
		#print(i)
X = np.asarray(X)
y = np.asarray(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
nx, nxx, nyx = X_train.shape
#ny, nxy, nyy = y_train.shape
X_train = np.nan_to_num(X_train)
X_train = X_train.reshape((nx,nxx*nyx))
#y_train = y_train.reshape((ny,nxy*nyy))
print(np.sum(X_train[1]),np.sum(X_train[2]),np.sum(X_train[3])) 
#print(X_train[1], X_train[100], X_train[500])
svm = SVC(kernel='linear')
print('TEST ', X_train[1].shape, nx, nxx, nyx)
svm.fit(X_train,y_train)

file = 'saved_model.sav'
#pickle.dump(svm, open(file,'wb'))

nTest, nxTest, nyTest = X_test.shape
X_test = X_test.reshape((nTest,nxTest*nyTest))
X_test = np.nan_to_num(X_test)
y_pred = svm.predict(X_test)

print(f1_score(y_test, y_pred, average="weighted"))


#print(len(kp))
#print(len(des))
#img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv.imwrite('sift_keypoints.jpg',img) # writes to a file
#plt.imshow(img)
#plt.show()

#question: 
#what does des variable from sift.compute() actually give?
#how to do dense sift?
