import numpy as np
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
plt.imshow(img)
plt.show()

#question: 
#what does des variable from sift.compute() actually give?
#how to do dense sift?
