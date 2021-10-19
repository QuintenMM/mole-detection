from PIL import Image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import os
import numpy as np
from sklearn.model_selection import train_test_split


def split_in_train_val_test(X, target):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        target,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    return X_train, X_val, y_train, y_val, X_test, y_test


def get_preprocessed_images_transfer_learning(images_directory: str, image_size: tuple, limit:int) -> np.array:
    images = []
    i = 0
    for img in os.listdir(images_directory):
        img = image.load_img(images_directory+img, target_size=image_size)
        img = image.img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        images.append(img)
        i += 1
        if i > limit:
            return np.vstack(images)
    return np.vstack(images)


def get_preprocessed_images_no_tl(images_directory: str, limit:int) -> np.array:
    images = []
    i = 0
    for img in os.listdir(images_directory):
        img = Image.open(images_directory+img)
        tf_image = np.array(img)
        img_resized = np.resize(tf_image, (224, 224, 3))
        img_resized = img_resized[:, :, ::-1]  # RGB to BGR
        img_reshaped = img_resized.reshape((1, img_resized.shape[0], img_resized.shape[1], img_resized.shape[2]))
        img_scaled = img_reshaped / 255
        images.append(img_scaled)
        i += 1
        if i > limit:
            return np.vstack(images)
    return np.vstack(images)


def prepare_one_image_no_tl(img):
    tf_image = np.array(img)
    img_resized = np.resize(tf_image, (224, 224, 3))
    # st.write(img_resized.shape)
    img_resized = img_resized[:, :, ::-1]  # RGB to BGR
    img_reshaped = img_resized.reshape(
        (1, img_resized.shape[0], img_resized.shape[1], img_resized.shape[2]))
    # st.write(img_reshaped.shape)
    img_scaled = img_reshaped / 255
    # st.write(img_scaled)
    return img_scaled
