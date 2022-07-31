import numpy as np
import os
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Model
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.utils import image_dataset_from_directory
from PIL import Image


def createVGG19():
    bm = VGG19(weights='imagenet')
    model = Model(inputs=bm.input, outputs=bm.get_layer('fc1').output)
    return model


def createNeighbors(vectors):
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(vectors)
    return knn


def preprocess_image(image):
    image_resized = image.resize((224, 224), resample=Image.Resampling.BILINEAR)
    img_array = np.array(image_resized)

    preprocessed_img = preprocess_input(img_array)
    preprocessed_img = np.expand_dims(preprocessed_img, 0)

    return preprocessed_img


def return_filenames(indices, filename='fnames'):
    with open(filename, 'r') as f:
        fnames = np.array(f.read().splitlines())

    return fnames[indices[0]]


def create_vectors(model, directory='images'):
    dataset = image_dataset_from_directory(
        directory=directory,
        label_mode=None,
        batch_size=1,
        image_size=(224, 224),
        shuffle=False
    )

    vectors = model.predict(dataset)
    return vectors


def create_file_list(directory='images'):
    return os.listdir(directory)
