import cv2
import sys
import numpy as np
from keras.models import load_model

def ToFace(filename, required_size=(256, 256)):
    sketch = cv2.resize(cv2.imread(filename),required_size)
    sketch = (sketch - 127.5) / 127.5
    model = load_model('/home/azureuser/gmodel2.h5')
    face = model.predict(np.array([sketch]))
    face = 127.5 * face[0] + 127.5
    face = cv2.cvtColor(face.astype('uint8'), cv2.COLOR_RGB2BGR)
    return face

res = ToFace(sys.argv[1])
cv2.imwrite(sys.argv[2],res)    
cv2.imwrite(sys.argv[3],res)
