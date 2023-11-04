import streamlit as st
import os
import cv2
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from scipy.spatial import distance
from PIL import Image
import numpy as np

embedding_model = tf.keras.models.load_model('Visual_Similarity_for_Search/Assets/embedding_model.h5')
transfer_model = tf.keras.models.load_model('Visual_Similarity_for_Search/Assets/transfer_model.h5')

tsne_embeddings = np.load('Visual_Similarity_for_Search/Assets/tsne_embeddings.npy')
image_embeddings = np.load('Visual_Similarity_for_Search/Assets/image_embeddings.npy')


files = os.listdir('Visual_Similarity_for_Search/Assets/images_original')
file_dict = {index: filename for index, filename in enumerate(files)}

dropdown = ['None']
dropdown = dropdown + list(file_dict.values())

st.title('Visual Similarity for Search')

st.write('## Select an image from the dropdown or upload a new image')

selected_value = st.selectbox("Select an image", dropdown)

if not selected_value == 'None':
    base_image = selected_value

    base_image_index = -1
    for i,j in file_dict.items():
        if j == base_image:
            base_image_index = i

    n_neighbors = st.number_input("Enter number of similar images to select", min_value = 1, max_value = 5)
    base_embedding = tsne_embeddings[base_image_index]
    distances = {i:distance.euclidean(base_embedding, embedding) for i,embedding in enumerate(tsne_embeddings)}

    nearest_neighbor_indices = np.argsort(list(distances.values()))[:n_neighbors + 1]

    nearest_neighbor_distances = []

    for i in nearest_neighbor_indices:
        nearest_neighbor_distances.append(distances[i])

    image_paths = [f'Visual_Similarity_for_Search/Assets/images_original/{file_dict[i]}' for i in nearest_neighbor_indices]

    st.write("## Base Image")
    st.image(image_paths[0], width = 300)

    st.write("## Similar Images")
    for i in range(1, len(image_paths)):
        st.image(image_paths[i], width = 300)
else:
    uploaded_file = st.file_uploader("Upload an image", type = ["jpg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption = "Uploaded image", use_column_width = True)

        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.image.resize(image, (224, 224))
        image = tf.expand_dims(image, 0)
        pred = transfer_model.predict(image)
        
        classes = ['Dress','Hat','Longsleeve','Shoes','T-Shirt']
        index_of_1 = np.argmax(pred)
        predicted_class = classes[int(index_of_1)]

        st.write("### Predicted value is :blue[" + predicted_class + "]")

        image_embedding = embedding_model.predict(image)

        tsne = TSNE(n_components=2, init='pca', perplexity=20, random_state=0)
        tsne_embeddings = tsne.fit_transform(np.vstack([image_embeddings, image_embedding]))

        n_neighbors = st.number_input("Enter number of similar images to select", min_value = 1, max_value = 5)
        base_embedding = tsne_embeddings[-1]
        distances = {i:distance.euclidean(base_embedding, embedding) for i,embedding in enumerate(tsne_embeddings)}

        nearest_neighbor_indices = np.argsort(list(distances.values()))[:n_neighbors + 1]

        nearest_neighbor_distances = []

        for i in nearest_neighbor_indices:
            nearest_neighbor_distances.append(distances[i])

        image_paths = [f'Visual_Similarity_for_Search/Assets/images_original/{file_dict[i]}' for i in nearest_neighbor_indices]

        st.write("## Similar Images")
        for i in range(1, len(image_paths)):
            st.image(image_paths[i], width = 300)
        
