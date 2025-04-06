from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input, concatenate
from tensorflow.keras.models import Model
import cv2
from pydantic import BaseModel
import python_multipart
from typing import Annotated, List
from os import path

IMG_SIZE = (256, 256)
MODEL_PATH = path.join("models", "unet_v8_multiclass_epoch_20.weights.h5")

class Files(BaseModel):
    pre_disaster_image: bytes
    post_disaster_image: bytes

def process_image(image: bytes):
    file_byte = np.array(bytearray(image), dtype=np.uint8)
    opencv = cv2.imdecode(file_byte, 1)
    opencv = cv2.cvtColor(opencv, cv2.COLOR_BGR2RGB)
    opencv = cv2.resize(opencv, IMG_SIZE) / 255.0
    return opencv

def overlay_masks_on_image(image, mask, alpha=0.4):
    """
    Overlays a color image with a 5-channel mask.
    
    Parameters:
        image (np.ndarray): RGB image of shape (X, Y, 3), dtype=uint8.
        mask (np.ndarray): One-hot encoded mask of shape (X, Y, 5), dtype=uint8 or bool.
        alpha (float): Transparency of the mask overlay.
        
    Returns:
        np.ndarray: Image of shape (X, Y, 3) with mask overlay.
    """
    # Define RGBA colors for each layer (skip layer 0: transparent)

    colors = {
        1: (0, 255, 255),    # Cyan
        2: (255, 255, 0),    # Yellow
        3: (255, 165, 0),    # Orange
        4: (255, 0, 0),      # Red
    }

    overlay = image.copy()

    for i in range(1, 5):  # Skip channel 0 (transparent)
        mask_layer = mask[:, :, i]
        color = np.array(colors[i], dtype=np.uint8)

        # Create a colored layer with the mask
        color_mask = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            color_mask[:, :, c] = color[c] * mask_layer

        # Blend the color mask with the original image
        overlay = np.where(mask_layer[..., None], 
                           (1 - alpha) * overlay + alpha * color_mask,
                           overlay)

    return overlay.astype(np.uint8)

def dataset_generator(pre_image, post_image):
    stacked_image = np.concatenate([pre_image, post_image], axis=-1)  # Shape: (256, 256, 6)
    yield stacked_image

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

app = FastAPI()

@app.get("/")
async def say_hi():
    return {"message": "Hello World!"}


# Endpoint pour recevoir des fichiers - Essai 1
@app.post("/uploadfiles/")
async def upload_files(request: Request):
    form = await request.form()
    pre_disaster_image = form['pre_disaster_image']
    post_disaster_image = form['post_disaster_image']

    # Convert the pre_disaster_image string to bytes
    pre_disaster_byte = pre_disaster_image.encode('utf-8')
    pre_file_byte = np.array(bytearray(pre_disaster_byte), dtype=np.uint8)
    pre_opencv = cv2.imdecode(pre_file_byte, 1)
    pre_opencv = cv2.cvtColor(pre_opencv, cv2.COLOR_BGR2RGB)
    pre_opencv = cv2.resize(pre_opencv, IMG_SIZE) / 255.0

    return {"pre_size": str(type(pre_disaster_image)), 
            "post_size": str(type(post_disaster_image))}


# Endpoint pour recevoir des fichiers - Essai 2
@app.post("/testfiles/")
async def test_files(files: List[UploadFile] = File(...)):
    pre_disaster_image = files[0].file.read()
    post_disaster_image = files[1].file.read()

    pre_processed = process_image(pre_disaster_image)
    post_processed = process_image(post_disaster_image)

    # stacked_image = np.concatenate([pre_processed, post_processed], axis=-1)  # Shape: (256, 256, 6)

    # Add stacked image to tf.data.Dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(pre_disaster_image, post_disaster_image), 
        output_signature=(tf.TensorSpec(shape=(256, 256, 6), dtype=tf.float32))
    )
    
    model = unet_model()
    model.load_weights(MODEL_PATH)

    pred = model.predict(tf.expand_dims(dataset.take(1)[1, :, :, :], axis=0))

    # return {"stacked_shape": stacked_image.shape}
    return FileResponse(pred)

# Endpoint pour recevoir l'image et renvoyer le résultat
@app.post("/predict")
async def predict(data: Files):
    print("\n1 - Received data\n")
    
    files = data.model_dump()
    
    print(f"\n2 - Dictionnary created with keys: {files.keys()}")
    print(f"2 - Dictionary value types: {type(files['pre_disaster_image'])}")
    print(f"2 - Dictionary value length: {len(files['pre_disaster_image'])}\n")

    # Pre-processing of received files
    pre_disaster_image = files['pre_disaster_image'].encode('utf-8')
    pre_file_byte = np.array(bytearray(pre_disaster_image), dtype=np.uint8)
    pre_opencv = cv2.imdecode(pre_file_byte, 1)
    pre_opencv = cv2.cvtColor(pre_opencv, cv2.COLOR_BGR2RGB)
    pre_opencv = cv2.resize(pre_opencv, IMG_SIZE) / 255.0

    post_disaster_image = files['post_disaster_image'].encode('utf-8')
    post_file_byte = np.array(bytearray(post_disaster_image), dtype=np.uint8)
    post_opencv = cv2.imdecode(post_file_byte, 1)
    post_opencv = cv2.cvtColor(post_opencv, cv2.COLOR_BGR2RGB)
    post_opencv = cv2.resize(post_opencv, IMG_SIZE) / 255.0

    stacked_image = np.concatenate([pre_opencv, post_opencv], axis=-1)  # Shape: (256, 256, 6)

    # model = # Model load

    # pred = model.predict(stacked_image)
    pred = ''
    
    return {"prediction": pred} # pred