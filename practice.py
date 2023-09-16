import cv2
from tensorflow.keras.utils import img_to_array
import numpy as np

image = cv2.imread('/Users/apple/Desktop/MC Project/flask_app/peachtest.jpg')
image = cv2.resize(image, (264,264)) 
print(image)  
image_array = img_to_array(image)
print(image_array)
image_array = np.array(image_array/255.0)
print(image_array)
image_array = np.expand_dims(image_array, axis=0)
print(image_array)