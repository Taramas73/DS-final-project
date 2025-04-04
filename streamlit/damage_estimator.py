import streamlit as st
import numpy as np
import cv2
import requests

st.title("Damage estimator")

st.text("Load the couple of images of the disaster photo you want to load")

# Essayer de mettre les images sur deux colonnes
# Un message de patiente 

col1, col2 = st.columns(2)

pre_disaster_file = col1.file_uploader("Choose a pre-disaster image file", type="png")
post_disaster_file = col2.file_uploader("Choose a post-disaster image file", type="png")

if pre_disaster_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(pre_disaster_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    col1.image(opencv_image, channels="BGR")

if post_disaster_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(post_disaster_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    col2.image(opencv_image, channels="BGR")

if st.button("Predict"):
    # Convert the pre-disaster image to bytes
    pre_disaster_image_bytes = pre_disaster_file.read()
    post_disaster_image_bytes = post_disaster_file.read()

    # Send a POST request to the API
    url = "http://localhost:8080/predict"  # Replace with your API endpoint
    files = {
        "pre_disaster_image": ("pre_disaster.png", pre_disaster_image_bytes, "image/png"),
        "post_disaster_image": ("post_disaster.png", post_disaster_image_bytes, "image/png"),
    }
    
    response = requests.post(url, files=files)
    

    if response.status_code == 200:
        st.success("Prediction successful!")
        st.json(response.json())
    else:
        st.error("Prediction failed.")
        st.error(f"Error: {response.status_code} - {response.text}")