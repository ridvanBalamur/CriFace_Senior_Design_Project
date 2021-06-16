import sys
import cv2

def ToSketch(filename, required_size=(256, 256)):
    canvas = cv2.imread(sys.argv[3],cv2.CV_8UC1)
    canvas = cv2.resize(canvas, required_size)
    gray = cv2.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE),required_size)
    inv = 255-gray
    blurred = cv2.GaussianBlur(inv, ksize=(21, 21), sigmaX=0, sigmaY=0)
    sketch = cv2.divide(gray,255-blurred,scale = 256)
    sketch = cv2.multiply(sketch, canvas, scale=1./ 256)
    sketch =  cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    return sketch

cv2.imwrite(sys.argv[2],ToSketch(sys.argv[1]))    
    