import pickle
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC # This is the SVM classifier
from matplotlib import pyplot as plt
import glob
import pickle
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def chunks(l, n):
    chunks = []
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        chunks.append(l[i:i + n])
    return chunks

def reshape_image(filePath):
  desired_size = 250
  im = filePath#cv.imread(filePath)
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

 
svm = pickle.load(open('saved_model.sav','rb'))
#img = cv.imread('../images/UFPR05/Sunny/2013-02-24/2013-02-24_06_20_00.jpg')
img = cv.imread('../images/UFPR05/Sunny/2013-02-28/2013-02-28_18_25_45.jpg')
#xSegment = #[555.75,746.25,714.75,534.75,498.75,671.25,644.25,483.75,452.25,606.75,579.75,437.25,411.75,555.75,528.75,395.25,381.75,506.25,482.25,366.75,353.25,474.75,443.25,335.25,324.75,438.75,411.75,312.75,308.25,404.25,381.75,293.25,281.25,380.25,351.75,267.75,267.75,356.25,332.25,252.75]
#ySegment = [536.75,	589.25,	662.75,	602.75,	449.75,	494.75,	554.75,	518.75,	377.75,	413.75,	467.75, 433.25,	317.75,	346.25,	392.75,	367.25,	259.25,	284.75,	#326.75,	305.75,	212.75,	235.25,	271.25,	254.75,	169.25,	190.25,	221.75,	206.75,	130.25,	145.25,	176.75,	166.25,	97.250,	112.25,	136.25,	130.25,	64.25,	#79.25,	100.25,	94.25]

xSegment = [744, 717, 607, 638, 669, 642, 555, 582, 610, 579, 503, 530, 557, 528, 456, 483, 509, 480, 417, 446, 473, 443, 387, 412, 438, 411, 354, 381, 402, 380, 332, 353, 379, 354, 309, 332, 357, 333, 284, 304]
ySegment = [588, 657, 624, 562, 495, 554, 533, 473, 413, 465, 450, 400, 345, 390, 380, 336, 287, 325, 314, 277, 235, 269, 255, 227, 190, 221, 210, 180, 147, 174, 167, 141, 110, 134, 127, 105, 79, 100, 92, 72]
x2ndrow = [746, 834, 823, 724, 678, 763, 748, 654, 623, 702, 680, 597, 575, 646, 629, 546, 530, 602, 579, 506, 494, 554, 535, 472, 461, 516, 497, 439, 429, 479, 465, 411, 402, 449, 433, 379, 379, 424, 406, 357]
y2ndrow = [488, 508, 575, 547, 407, 426, 485, 465, 338, 354, 403, 388, 285, 297, 336, 323, 233, 245, 279, 266, 191, 197, 228, 216, 148, 157, 183, 173, 111, 120, 143, 132, 78, 85, 106, 98, 49, 56, 73, 66]
x3rdrow = [1090, 1171, 1173, 1089, 1010, 1083, 1081, 1009, 942, 1006, 997, 933, 875, 938, 929, 867, 821, 881, 864, 811, 775, 825, 807, 764, 729, 774, 760, 714, 691, 734, 715, 676, 649, 693, 676, 630]
y3throw = [350, 364, 418, 405, 288, 303, 350, 340, 240, 251, 291, 279, 196, 205, 239, 229, 152, 160, 195, 184, 118, 124, 154, 147, 84, 89, 117, 111, 56, 60, 83, 79]
x4throw = [1146, 1238, 1234, 1144, 1064, 1154, 1152, 1062, 995, 1077, 1074, 988, 933, 998, 993, 926, 876, 949, 945, 868, 825, 885, 880, 817, 778, 839, 834, 771, 740, 788, 782, 733, 703, 754, 745, 692]
y4throw = [300, 314, 355, 340, 248, 264, 293, 283, 198, 209, 242, 229, 157, 168, 194, 181, 121, 130, 152, 141, 89, 95, 116, 108, 59, 64, 82, 76, 31, 37, 52, 45, 4, 8, 26, 19]
x5throw = [1188, 1251, 1254, 1192, 1123, 1180, 1180, 1125, 1065, 1115, 1115, 1062, 1009, 1059, 1052, 1002, 956, 1003, 996, 950]
y5throw = [138, 144, 177, 172, 105, 108, 138, 134, 71, 78, 106, 100, 43, 48, 77, 70, 16, 21, 49, 44]



coordinates = []
for x,y in zip(xSegment, ySegment):
    coordinates.append((x,y))
betterCoordinates = []
print(betterCoordinates)
 
betterCoordinates = chunks(coordinates,4)
betterCoordinates = np.array(betterCoordinates)
#betterCoordinates = np.rint(betterCoordinates)
print(betterCoordinates[1])
i = 0
for filename in glob.glob('../images/empty/*'):
	if(i < 0):
		
		#img = cv.imread(filename)
		#gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
		filename = cv.imread(filename)
		img = reshape_image(filename)
		cv.imshow('img',img)
		cv.waitKey(0)
		des = dsift(img)
		nx, ny = des.shape
		des = np.nan_to_num(des)
		des = des.reshape((nx*ny))
		#if(i == 0):
			#plt.imshow(img, cmap='gray')
			#plt.show()
			#print(des)
			
		print(svm.predict(des.reshape(1,-1)))
		i = i + 1
		#print(des)
for i in range(0,len(betterCoordinates)):
	pts = betterCoordinates[i]
	#pts = pts.astype(int)	
	print(pts)
	pts = np.array(pts)
	boundBox = cv.boundingRect(pts)
	x,y,w,h = boundBox
	croped = img[y:y+h, x:x+w].copy()
	print(croped.shape)
	inputImg = reshape_image(croped)
	inputImg = np.rot90(inputImg)
	cv.imshow('idk',inputImg)
	cv.waitKey(0)
	inputDes = dsift(inputImg)
	nx, ny = inputDes.shape
	inputDes = np.nan_to_num(inputDes)
	inputDes = inputDes.reshape((nx*ny))
	print (inputDes.shape)
	print(svm.predict(inputDes.reshape(1,-1)))


pts = np.array([[555, 536],
 [746, 589],
 [714, 662],
 [534, 602]])

boundBox = cv.boundingRect(pts)
x,y,w,h = boundBox
croped = img[y:y+h, x:x+w].copy()


inputImg = reshape_image(croped)
cv.imwrite('idk.png',inputImg)
inputDes = dsift(inputImg)
nx, ny = inputDes.shape
inputDes = np.nan_to_num(inputDes)
inputDes = inputDes.reshape((nx*ny))
print (inputDes.shape)
print(svm.predict(inputDes.reshape(1,-1)))


