import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image  # For loading and displaying images

# Load the pre-trained model
model = load_model(r"D:\All_DataSet2\Trafic Sign recognition\trafic.h5")

# Define labels for prediction
labels = [
    "Ahead only", "Beware of ice/snow", "Bicycles crossing", "Bumpy road", "Children crossing", 
    "Dangerous curve left", "Dangerous curve right", "Double curve", "End no passing", 
    "End of no passing", "End of speed limit (80km/h)", "End speed + passing limits", 
    "General caution", "Go straight or left", "Go straight or right", "Keep left", "Keep right", 
    "No entry", "No passing", "No passing vehicles over 3.5 tons", "No vehicles", "Pedestrians", 
    "Priority road", "Right-of-way at intersection", "Road narrows on the right", "Road work", 
    "Roundabout mandatory", "Slippery road", "Speed limit (20km/h)", "Speed limit (30km/h)", 
    "Speed limit (50km/h)", "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)", 
    "Speed limit (100km/h)", "Speed limit (120km/h)", "Stop", "Traffic signals", 
    "Turn left ahead", "Turn right ahead", "Vehicles over 3.5 tons prohibited", 
    "Wild animals crossing", "Yield"
]

# Streamlit app
st.title("Traffic Sign Prediction")
st.write("Upload an image of a traffic sign to predict its category.")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = Image.open(uploaded_file)
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 150, 150, 3)  # Reshape to match model input

    # Make prediction
    predictions = model.predict(img_array)
    predicted_index = predictions.argmax()
    predicted_label = labels[predicted_index]

    # Display the result
    st.write(f"**Predicted Label:** {predicted_label}")
