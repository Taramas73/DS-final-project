import streamlit as st
import numpy as np
import cv2
import requests
import chardet
from os import path, environ

pre_bytes = None
post_bytes = None

st.title("Damage estimator")

st.text("Load the couple of images of the disaster photo you want to load")

tab1, tab2 = st.tabs(["Load from file", "Load from URL"])

# Tab 1 - FROM FILE
col1, col2 = tab1.columns(2)

pre_disaster_file = col1.file_uploader("Choose a pre-disaster image file", type="png")
post_disaster_file = col2.file_uploader("Choose a post-disaster image file", type="png")

if pre_disaster_file is not None:
    # Convert the file to an opencv image.
    pre_bytes = pre_disaster_file.read()
    file_bytes = np.asarray(bytearray(pre_bytes), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    # Display image.
    col1.image(opencv_image, channels="BGR")

if post_disaster_file is not None:
    # Convert the file to an opencv image.
    post_bytes = post_disaster_file.read()
    file_bytes = np.asarray(bytearray(post_bytes), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    # Display image.
    col2.image(opencv_image, channels="BGR")

# Tab2 - FROM URL
col3, col4 = tab2.columns(2)

pre_disaster_url = col3.text_input("Enter the URL of the pre-disaster image")
post_disaster_url = col4.text_input("Enter the URL of the post-disaster image")

if pre_disaster_url:
    file_response = requests.get(pre_disaster_url)
    col3.image(file_response.content, channels="BGR")

if post_disaster_url:
    file_response = requests.get(post_disaster_url)
    col4.image(file_response.content, channels="BGR")

# Prédiction
if tab1.button("Predict from file"):
    # Send a POST request to the API
    files = [('files', pre_bytes), ('files', post_bytes)]
    
    response = requests.post(environ.get('PRED_URL'), files=files)
    
    if response.status_code == 200:
        tab1.success("Prediction successful!")
        mask = response.content
        tab1.image(mask)
        # tab1.write(str(type(mask)))
    else:
        tab1.error("Prediction failed.")
        with open(path.join("logs", "response.txt"), "w") as f:
            f.write(str(response.status_code) + "\n\n" + response.text)
        # tab1.error(f"Error: {response.status_code} - {response.text}")

if tab2.button("Predict from URL"):
    # Convert both images to bytes
    pre_response = requests.get(pre_disaster_url)
    # tab2.write(pre_response.headers)
    pre_disaster_image_bytes = pre_response.content
    post_response = requests.get(post_disaster_url)
    post_disaster_image_bytes = post_response.content

    # Send a POST request to the API
    files = [('files', pre_disaster_image_bytes), ('files', post_disaster_image_bytes)]
    
    response = requests.post(environ.get('PRED_URL'), files=files)
    
    if response.status_code == 200:
        tab2.success("Prediction successful!")
        mask = response.content
        tab2.image(mask)
    else:
        tab2.error("Prediction failed.")
        with open(path.join("logs", "response.txt"), "w") as f:
            f.write(str(response.status_code) + "\n\n" + response.text)
        # tab2.error(f"Error: {response.status_code} - {response.text}")