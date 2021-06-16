import cv2
from mtcnn.mtcnn import MTCNN
import sys

def ToJpg(filename,savename):
	im = cv2.imread(filename)
	cv2.imwrite(savename,im)
	im_jpg = cv2.imread(savename)
	return im_jpg

def ExistFace(photo):
	detector = MTCNN()
	results = detector.detect_faces(photo)
	if len(results) == 0:
		return 0
	return 1
 

print(ExistFace(ToJpg(sys.argv[1], sys.argv[2])))