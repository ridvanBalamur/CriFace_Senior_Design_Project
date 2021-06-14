from matplotlib import pyplot
import cv2
import numpy as np
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import sys

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = pyplot.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[20:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array
 
# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
	# extract faces
	faces = [extract_face(f) for f in filenames]
	# convert into an array of samples
	samples = asarray(faces, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples, version=2)
	# create a vggface model
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	# perform prediction
	yhat = model.predict(samples)
	return yhat
 
# determine if a candidate face is a match for a known face
def similarity(known_embedding, candidate_embedding):
	# calculate distance between embeddings
	score = cosine(known_embedding, candidate_embedding)
	return 100*(1-score)


attributes = []
folder = sys.argv[1]
for i in range(8):
	attributes.append(folder+str(i)+".jpg")
embeddings = get_embeddings(attributes)
suspect = get_embeddings([folder + 'suspect.jpg'])

scores = [similarity(suspect,e) for e in embeddings]
max_value = max(scores)
max_index = scores.index(max_value)
best_photo = attributes[max_index]
best_match = cv2.imread(best_photo)
cv2.imwrite(sys.argv[2], best_match)
if max_value < 50 :
	message = "Not the same person"
else:
	message = '%' + str(int(np.round(max_value)))
print(message)