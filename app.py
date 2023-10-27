import streamlit as st

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
print(tf.__version__)

import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 'truetype'})
from matplotlib import offsetbox
import numpy as np
from tqdm import tqdm

import glob
import ntpath
import cv2

from sklearn.metrics.pairwise import cosine_similarity
from sklearn import manifold
import scipy as sc


#Loading Images
image_paths = glob.glob('/workspaces/Assignment03/Assets/Images/*.jpg')
print(f'Found [{len(image_paths)}] images')
images = {}
for image_path in image_paths:
                image = cv2.imread(image_path, 3)
                b,g,r = cv2.split(image)           # get b, g, r
                image = cv2.merge([r,g,b])         # switch it to r, g, b
                image = cv2.resize(image, (200, 200))
                images[ntpath.basename(image_path)] = image 

if st.sidebar.button('Run'):
        try:    
            st.write("Images Available")             
            n_col = 8
            n_row = int(len(images) / n_col)
            
            #f, ax = plt.subplots(n_row, n_col, figsize=(16, 8))
            for i in range(n_row):
              for j in range(n_col):
                index = n_col * i + j
                if index < len(images):
                    st.image(
                    images[list(images.keys())[index]],
                    caption=list(images.keys())[index],
                    width=100 )              
        except Exception as e:
          st.error(f"Error executing the query: {str(e)}")
