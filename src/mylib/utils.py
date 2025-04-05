import os
import cv2
import numpy as np
import tensorflow as tf

# Function to load and preprocess images and masks
def load_image(image_path,verbose=0, IMG_SIZE=(256, 256)):
    """Load and preprocess images."""
    img = cv2.imread(image_path)

    if verbose==0:  # LOad messages
        pass
    elif verbose == 1:
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return None
        print(f"✅ Loaded image {image_path}, Shape: {img.shape}, Dtype: {img.dtype}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE) / 255.0  # Normalize
    return img

def load_mask(mask_path,num_classes=5,verbose=0, IMG_SIZE=(256, 256)):
    """Load and preprocess masks."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if verbose==0:  # LOad messages
        pass
    elif verbose == 1:
        if mask is None:
            print(f"❌ Failed to load mask: {mask_path}")
            return None
        print(f"✅ Loaded mask {mask_path}, Shape: {mask.shape}, Dtype: {mask.dtype}")

    mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    mask = tf.keras.utils.to_categorical(mask, num_classes=num_classes)
    mask.astype(np.float32)
    return mask

# Find images with at least X unique pixel values
def find_images_with_min_unique_values(folder_path, min_unique_values=3):
    """Finds images in a folder with at least `min_unique_values` unique pixel values."""
    
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    valid_images = []

    for file in image_files:
        print(f"Checking {file}...")
        img_path = os.path.join(folder_path, file)
        mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        
        if mask is None:
            print(f"❌ Could not load {file}")
            continue
        
        unique_values = np.unique(mask)  # Get unique pixel values
        
        if len(unique_values) >= min_unique_values:
            valid_images.append(file)  # Store filenames that meet the condition

    return valid_images

# # Example usage
# folder = "../dataset/train/targets"
# matching_images = find_images_with_min_unique_values(folder, min_unique_values=5)

# print("Images with at least 3 unique pixel values:")
# matching_images

# Function to load dataset using tf.data
def dataset_generator(image_filenames, images_path, masks_path):
    """Generator function to yield images and masks."""
    for filename in image_filenames:
        pre_path = os.path.join(images_path, filename)
        post_path = os.path.join(images_path, filename.replace("_pre_disaster", "_post_disaster"))
        mask_path = os.path.join(masks_path, filename.replace("_pre_disaster.png", "_post_disaster_target.png"))

        pre_img = load_image(pre_path)
        post_img = load_image(post_path)
        mask_img = load_mask(mask_path) # Now returns (256, 256, 5)


        if pre_img is not None and post_img is not None and mask_img is not None:
            stacked_image = np.concatenate([pre_img, post_img], axis=-1)  # Shape: (256, 256, 6)
            yield stacked_image, mask_img

# Create TensorFlow dataset
def get_tf_dataset(image_filenames, images_path, masks_path, batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(image_filenames, images_path, masks_path),
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 6), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256, 5), dtype=tf.float32) # One-hot encoded mask
        )
    )

    dataset = dataset.shuffle(BUFFER_SIZE)  # Shuffle the data
    dataset = dataset.batch(batch_size)  # Load in small batches
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Optimize loading

    return dataset


