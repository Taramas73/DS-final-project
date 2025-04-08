import streamlit as st
import numpy as np
import cv2
import requests
from os import path, environ
import pandas as pd

pre_bytes = None
post_bytes = None

labels = {'Severity': ['No damage', 'Minor damage', 'Major damage', 'Destroyed']}
# labels = {'s1': ['No damage'], 's2': ['Minor damage'], 's3': ['Major damage'], 's4': ['Destroyed']}
df = pd.DataFrame(labels)
df = df.style.map(lambda x: f"background-color: {'cyan' if 'No' in x else ('yellow' if 'Min' in x else ('orange' if 'Maj' in x else 'red')) }")

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
if col1.button("Predict from file"):
    # Send a POST request to the API
    files = [('files', pre_bytes), ('files', post_bytes)]
    
    response = requests.post(st.secrets['PRED_URL'], files=files)
    
    if response.status_code == 200:
        col5, col6= tab1.columns(2)
        col5.success("Prediction")
        mask = response.content
        col5.image(mask)

        m_resp = requests.post(st.secrets['PRED_MASK'], files=files)
        if m_resp.status_code == 200:
            col6.success("Mask")
            mask = m_resp.content
            col6.image(mask)
            col9, col10, col11 = tab1.columns(3)
            col10.write("Labels:")
            col10.dataframe(df, hide_index=True)
        else:
            col6.error("Mask cannot be loaded.")
    else:
        tab1.error("Prediction failed.")
        # with open(path.join("logs", "response.txt"), "w") as f:
        #     f.write(str(response.status_code) + "\n\n" + response.text)
        tab1.error(f"Error: {response.status_code} - {response.text}")

if tab2.button("Predict from URL"):
    # Convert both images to bytes
    pre_response = requests.get(pre_disaster_url)
    # tab2.write(pre_response.headers)
    pre_disaster_image_bytes = pre_response.content
    post_response = requests.get(post_disaster_url)
    post_disaster_image_bytes = post_response.content

    # Send a POST request to the API
    files = [('files', pre_disaster_image_bytes), ('files', post_disaster_image_bytes)]
    
    response = requests.post(st.secrets['PRED_URL'], files=files)
    
    if response.status_code == 200:
        col7, col8 = tab2.columns(2)
        col7.success("Prediction")
        mask = response.content
        col7.image(mask)

        m_resp = requests.post(st.secrets['PRED_MASK'], files=files)
        if m_resp.status_code == 200:
            col8.success("Mask")
            mask = m_resp.content
            col8.image(mask)
            col12, col13, col14 = tab2.columns(3)
            col13.write("Labels:")
            col13.dataframe(df, hide_index=True)
        else:
            col8.error("Mask cannot be loaded.")
    else:
        tab2.error("Prediction failed.")
        # with open(path.join("logs", "response.txt"), "w") as f:
        #     f.write(str(response.status_code) + "\n\n" + response.text)
        tab2.error(f"Error: {response.status_code} - {response.text}")