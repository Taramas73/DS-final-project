import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


# -------------------- TITLE --------------------
st.title("🧠 Model Development & Training")
st.markdown("---")

# -------------------- 1. OVERVIEW --------------------
st.header("1. Overview")
st.markdown("""
This section presents the deep learning model developed to assess building damage using pairs of satellite images captured before and after a disaster.  
The objective was to create a robust segmentation model capable of identifying and classifying damaged structures across various disaster scenarios.
""")

# -------------------- 2. MODEL ARCHITECTURE --------------------
st.header("2. Model Architecture")
st.markdown("""
U-Net is a convolutional neural network (CNN) architecture designed for semantic 
            segmentation, especially in tasks where the output is a pixel-wise 
            classification (e.g., medical images, satellite imagery).
            """)

# Placeholder for architecture image
st.image("data/unet.png", caption="U-Net Architecture Diagram (Example)")

with st.expander("Show model code snippet"):
    st.code("""
def unet_model(input_shape=(X_SIZE, Y_SIZE, 6), num_classes=5):
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

    """, language="python")

# -------------------- 3. DATA PREPARATION --------------------
st.header("3. Dataset Preparation & Preprocessing")
st.markdown("""
The **xView2 dataset** was used, consisting of high-resolution satellite imagery and corresponding segmentation masks annotated with four damage classes:
- No damage
- Minor damage
- Major damage
- Destroyed

Key preprocessing steps:
- Resized to 256×256 for more efficient training
- Stacked pre- and post-disaster images to create a 6-channel input (256x256x6)
- Normalized pixel values to [0, 1]
- Encoded masks to categorical format
- Applied data augmentation (random flips, rotations, brightness changes)
            
Key notes:
- The training set consisted of 2799 couples of pre- and post-disaster images and their corresponding masks.
- Masks that contained at least 3 categories were selected and augmented to ensure balanced representation of all classes.
- A total of 1146 masks met these criteria, resulting in 3945 data points overall.
- The validation set consisted of 1000 couples of pre- and post-disaster images.

""")

# -------------------- 4. TRAINING STRATEGY --------------------
st.header("4. Training Strategy")
st.markdown("""
The model was trained using **weighted categorical cross-entropy** criterion, designed to balance class imbalances and improve boundary segmentation.  
Weights of each class:
- Background: 0.1
- No Damage: 1.0
- Minor Damage: 1.2
- Major Damage: 1.2
- Destroyed: 1.2
Optimizer: **Adam**, Learning Rate: **1e-4**, Batch Size: **8** to handle big amount of data.
Number of epochs: **20** , Training time: **6.5 hours** on my CPU
""")

# Example of training curve (use your actual chart)
st.image("data/loss.png", caption="Training Loss Curve")
st.image("data/accuracy.png", caption="Training Accuracy Curve")
st.image("data/f1_score.png", caption="F1 Score Curve")


# -------------------- 5. EVALUATION & RESULTS --------------------
st.header("5. Evaluation & Results")
st.markdown("""
Model performance was evaluated using **F1 Score** and **accuracy** per damage class.  
The table below shows results on the validation set:
""")

# Performance Table
st.table({
    "Class": ["Background", "No Damage", "Minor Damage", "Major Damage", "Destroyed","Accuracy","Macro avg","Weighted avg"],
    "Precision": [0.99, 0.31, 0.04, 0.00, 0.00, None, 0.27, 0.94],
    "Recall": [0.94, 0.76, 0.01, 0.00, 0.00, None, 0.34, 0.91],
    "F1 Score": [0.96, 0.44, 0.01, 0.00, 0.00, 0.91, 0.28, 0.92],
    "Support": [2465005, 109312, 23041, 20637, 3445, 2621440, 2621440, 2621440]
})

# Side-by-side visuals
st.subheader("Visual Results")
col1, col2, col3, col4, col5 = st.columns(5)
col1.image("data/example1.png")
col2.image("data/example2.png")
col3.image("data/example3.png")
col4.image("data/example4.png")
col5.image("data/example5.png")

# -------------------- 6. CHALLENGES & FUTURE WORK --------------------
st.header("6. Challenges & Future Work")
st.markdown("""
**Challenges faced:**
- Class imbalance, especially for "major damage" and "destroyed" classes
- Visual ambiguity due to clouds, lighting, or minimal structural change
- Geographic variability in building style and density

**Future improvements:**
- Training on whole dataset (~6x more data)
- Training on unresized images (1024x1024)
- Implementing more complex models
""")



st.markdown("---")
st.success("📌 End of Model Development Section")

