# Import all the necessary libraries and modules
import torch
import numpy
import cv2
import streamlit as st
import torch.nn.functional as F
import torchvision.models as models

from torch import nn
from PIL import Image
from torchvision import transforms

st.title("Dog Breed Predictor")

def load_models_and_weights():
	# Load the saved weights and bias into our classifier
	dog_classifier.load_state_dict(torch.load('models/dog_breed_classifier.pt',
										   map_location=torch.device('cpu')))
	face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
	
	return face_cascade

def human_face_detector(img, face_cascade):
	img = numpy.asarray(img)
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	faces = face_cascade.detectMultiScale(gray)

	return True if len(faces) > 0 else False

# def detect_dog(img, dog_detector):
# 	torch.cuda.empty_cache()
# 	if use_cuda:
# 		dog_detector = dog_detector.cuda()

# 	input_batch = preprocess_image(img)

# 	with torch.no_grad():
# 		output = dog_detector(input_batch)

# 	_, index = torch.max(F.softmax(output[0], dim=0), 0)
# 	print(index)
# 	if index >= 151 and index <= 268:
# 		return True
# 	return False

def preprocess_image(img):
    input_img = Image.open(img)
    preprocess = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(input_img)
    input_batch = input_tensor.unsqueeze(0)

    # Move the input into the gpu if available
    if use_cuda:
        input_batch = input_batch.cuda()

    return input_batch

 # Function takes an image as an input and returns predicted dog breed
@st.cache
def predict_breed(img, dog_classifier):

    print("Predicting...\n")
    # load the image and return the predicted breed
    input_batch = preprocess_image(img)

    with torch.no_grad():
        output = dog_classifier(input_batch)

    prob, index = torch.max(F.softmax(output[0]), 0)
    prob = prob.cpu().numpy()

    return prob, dog_labels[index]



# Load the vgg16 model
dog_classifier = models.vgg16(pretrained=True)

# Freeze the weights
for param in dog_classifier.parameters():
    param.requires_grad = False

n_inputs = dog_classifier.classifier[6].in_features
n_classes = 133

# Specify the model architecture
dog_classifier.classifier[6] = nn.Sequential(
                                    nn.Linear(n_inputs, 1024),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(1024, n_classes)
                                )

# Use cuda (GPU) for processing
# Check if cuda is available
use_cuda = torch.cuda.is_available()
torch.cuda.empty_cache()
if use_cuda:
    dog_classifier = dog_classifier.cuda()

# Load the dog labels saved for prediction from the file
dog_labels = []

# open file and read contents in the list
with open('dog_labels.txt', 'r') as f:
    for line in f:
        # remove line break which is the last character in the string
        label = line[:-1]

        # add label to the list
        dog_labels.append(label)


# Image to predict the breed
uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:

	if st.button("Upload and Predict"):

	    face_cascade = load_models_and_weights()
	    image = Image.open(uploaded_image)
	    human_detected = human_face_detector(image, face_cascade)

	    if human_detected:
		    st.warning("Human Face Detected!")
	    st.write("Classifying...")
	    prob, label = predict_breed(uploaded_image, dog_classifier)
	    st.image(image, width=300, caption=label, use_column_width = True)
	    st.write("")

	    st.success("Predicted Breed: {}".format(label))
	    st.success("Probability: {:.2f}%".format(100 * prob))

