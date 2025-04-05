from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse, Response
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input, concatenate
from tensorflow.keras.models import Model
import cv2
from pydantic import BaseModel
from os import path

IMG_SIZE = (256, 256)
MODEL_PATH = path.join('..', 'models', 'unet_v8_multiclass_epoch_20.weights.h5')

# U-Net Model
def unet_model(input_shape=(256, 256, 6), num_classes=5):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)

    c2 = Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3,3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)

    c3 = Conv2D(256, (3,3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3,3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)

    # Bottleneck
    c4 = Conv2D(512, (3,3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3,3), activation='relu', padding='same')(c4)

    # Decoder
    u5 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, (3,3), activation='relu', padding='same')(u5)
    c5 = Conv2D(256, (3,3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, (3,3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3,3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3,3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3,3), activation='relu', padding='same')(c7)

    outputs = Conv2D(num_classes, (1,1), activation='softmax')(c7)  # Multiclass segmentation output
    model = Model(inputs, outputs)
    return model

class Files(BaseModel):
    pre_disaster_image: bytes
    post_disaster_image: bytes

app = FastAPI()

@app.get("/")
async def say_hi():
    return {"message": "Hello World!"}

# Endpoint pour recevoir l'image et renvoyer le résultat

@app.post("/predict")
async def predict(request: Request):
    data: bytes = await request.body()
    print(f"1 - Received data of length: {len(data)} bytes")
    stacked_bytes = np.array(bytearray(data), dtype=np.uint8)
    print(f"2 - Converted data to numpy array of shape: {stacked_bytes.shape}")
    stacked_image = cv2.imdecode(stacked_bytes, 1)
    print(f"3 - Decoded image shape: {stacked_image.shape}")
    stacked_image = cv2.cvtColor(stacked_image, cv2.COLOR_BGR2RGB)
    print(f"4 - Converted image to RGB shape: {stacked_image.shape}")
    stacked_image = cv2.resize(stacked_image, IMG_SIZE) / 255.0  # Normalize
    print(f"5 - Resized image shape: {stacked_image.shape}")

    # model = # Model load
    model = unet_model()
    model.load_weights(MODEL_PATH)
    # pred = model.predict(stacked_image)
    pred = model.predict(tf.expand_dims(stacked_image, axis=0))
    
    return FileResponse(pred) # pred