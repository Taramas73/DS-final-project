from fastapi import FastAPI, Request
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input, concatenate
from tensorflow.keras.models import Model
import cv2

IMG_SIZE = (256, 256)

app = FastAPI()

@app.get("/")
async def say_hi():
    return {"message": "Hello World!"}

# Endpoint pour recevoir l'image et renvoyer le résultat

@app.post("/predict")
async def predict(request: Request):
    data: bytes = await request.body()
    
    # Pre-processing of received files
    files = request.files
    pre_disaster_image = files['pre_disaster_image'].read()
    pre_file_byte = np.array(bytearray(pre_disaster_image), dtype=np.uint8)
    pre_opencv = cv2.imdecode(pre_file_byte, 1)
    pre_opencv = cv2.cvtColor(pre_opencv, cv2.COLOR_BGR2RGB)
    pre_opencv = cv2.resize(pre_opencv, IMG_SIZE) / 255.0

    post_disaster_image = files['post_disaster_image'].read()
    post_file_byte = np.array(bytearray(post_disaster_image), dtype=np.uint8)
    post_opencv = cv2.imdecode(post_file_byte, 1)
    post_opencv = cv2.cvtColor(post_opencv, cv2.COLOR_BGR2RGB)
    post_opencv = cv2.resize(post_opencv, IMG_SIZE) / 255.0

    stacked_image = np.concatenate([pre_opencv, post_opencv], axis=-1)  # Shape: (256, 256, 6)

    # model = # Model load

    # pred = model.predict(stacked_image)
    pred = ''
    
    return {"prediction": pred} # pred