import os
import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
import random
import warnings
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 'truetype'})
warnings.simplefilter("ignore")
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
ds = tfp.distributions


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

TRAIN_BUF = 60000
BATCH_SIZE = 512
DIMS = (28, 28, 1)
N_TRAIN_BATCHES = int(TRAIN_BUF/BATCH_SIZE)

# split dataset
train_images = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32") / 255.0
train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
    .shuffle(TRAIN_BUF)
    .batch(BATCH_SIZE)
)
# Load the style embeddings from the pickle file
with open('model_vae.pkl', 'rb') as file:
    loadeddata = pickle.load(file)

x_grid = loadeddata['x_grid']
embeddigns = loadeddata['embeddigns']

st.title("Visual Search Using Variational Autoeconders")
if st.sidebar.button('Load Images/Data'):
    # Define the text labels
    fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9

    fig, ax = plt.subplots(10, 11, figsize=(12, 10), gridspec_kw={'width_ratios': [2] + [1]*10})
    img_idx = 0
    for i in range(10):
        ax[i, 0].axis('off')
        ax[i, 0].text(0.5, 0.5, fashion_mnist_labels[i])
    
        class_indexes = [k for k, n in enumerate(y_train) if n == i]
        for j in range(10):
            ax[i, j+1].imshow(1 - train_images[class_indexes[j], :])
            ax[i, j+1].axis('off')
    st.pyplot(fig,use_container_width=True)


if st.sidebar.button('Generate Manifold Visualization'):

    #
    # Create a grid over the semantic space
    #
    nx = ny = 10
    meshgrid = np.meshgrid(np.linspace(-3, 3, nx), np.linspace(-3, 3, ny))
    meshgrid = np.array(meshgrid).reshape(2, nx*ny).T

    x_grid = x_grid.numpy().reshape(nx, ny, 28,28, 1)

        #
        # Visualize the reconstructed images
        #
    canvas = np.zeros((nx*28, ny*28))
    for xi in range(nx):
        for yi in range(ny):
            canvas[xi*28 : xi*28+28,  yi*28 : yi*28+28] = x_grid[xi, yi, :, :, :].squeeze()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(1 - canvas)
    ax.axis('off')
    st.pyplot(fig, use_container_width=True)


if st.sidebar.button('Find Similar Images'):
    query_image_id = 15
    k = 6


    def query(image_id, k):
        query_embedding = embeddigns[image_id]
        distances = np.zeros(len(embeddigns))
        for i, e in enumerate(embeddigns):
            distances[i] = np.linalg.norm(query_embedding - e)
        return np.argpartition(distances, k)[:k]

    idx = query(query_image_id, k=k)

    fig, ax = plt.subplots(1, k, figsize=(k*2, 2))
    for i in range(k):
        ax[i].imshow(1 - train_images[idx[i], :])
        ax[i].axis('off')
    
    plt.savefig('query-example-2.pdf')
    st.pyplot(fig, use_container_width=True)