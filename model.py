import keras
import cv2
import tensorflow as tf
import pickle
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import PIL
import base64
from io import BytesIO
from tensorflow.keras.utils import img_to_array

# Load your machine learning model here
model = keras.models.load_model("/Users/apple/Desktop/MC Project/ML Model/cnn.h5")

# Define a list of class labels for prediction
with open('/Users/apple/Desktop/MC Project/ML Model/label_binarizer.pkl', 'rb') as f:
    label_binarizer = pickle.load(f)
class_labels = list(label_binarizer.classes_)



    # Load and preprocess the image
image = cv2.imread('/Users/apple/Desktop/MC Project/ML Model/peachtest2.jpg')
image = cv2.resize(image, (264,264))   
image_array = img_to_array(image)
image_array = np.array(image_array/255.0)
image_array = np.expand_dims(image_array, axis=0)


    # Perform prediction
predictions = model.predict(image_array)
predicted_class_label = label_binarizer.inverse_transform(predictions)

print(predicted_class_label)


