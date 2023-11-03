import streamlit as st
import pickle

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


image_paths = glob.glob('./Assets/Images/*.jpg')
print(f'Found [{len(image_paths)}] images')

images = {}
image_style_embeddings = {}

for image_path in image_paths:
    image = cv2.imread(image_path, 3)
    b,g,r = cv2.split(image)           # get b, g, r
    image = cv2.merge([r,g,b])         # switch it to r, g, b
    image = cv2.resize(image, (200, 200))
    images[ntpath.basename(image_path)] = image      



# Load the style embeddings from the pickle file
with open('style_embeddings.pkl', 'rb') as file:
    image_style_embeddings = pickle.load(file)


st.sidebar.title('Image Search By an Artistic Style')
if st.sidebar.button('Run'):
        try:    
            st.write("Images Available")             
            n_col = 8
            n_row = int(len(images) / n_col)
            
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

if st.sidebar.button('Generate'):
#
# Visualize the 2D-projection of the embedding space with example images (thumbnails)
#

    try: 
      def embedding_plot(X, images, thumbnail_sparsity = 0.005, thumbnail_size = 0.3):
          x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
          X = (X - x_min) / (x_max - x_min)
          fig, ax = plt.subplots(1, figsize=(12, 12))

          shown_images = np.array([[1., 1.]])
        
          for i in range(X.shape[0]):
              if np.min(np.sum((X[i] - shown_images) ** 2, axis=1)) < thumbnail_sparsity: continue
              shown_images = np.r_[shown_images, [X[i]]]
              thumbnail = offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r, zoom=thumbnail_size)
              ax.add_artist(offsetbox.AnnotationBbox(thumbnail, X[i], bboxprops = dict(edgecolor='white'), pad=0.0))

          plt.grid(True)
      
  
      st.title("2D Projection of Embedding Space")
      st.write("Visualizing the 2D projection with example images (thumbnails)")
      tsne = manifold.TSNE(n_components=2, init='pca', perplexity=10, random_state=0)
      X_tsne = tsne.fit_transform( np.array(list(image_style_embeddings.values())) )
      fig= embedding_plot(X_tsne, images=list(images.values()))
      st.set_option('deprecation.showPyplotGlobalUse', False)
      st.pyplot(fig, use_container_width=True)
    
    except Exception as e:
          st.error(f"Error executing the query: {str(e)}")  

image_name_mapping = {
    "Cubism Style 1": "s_cubism-02.jpg",
    "Cubism Style 2": "s_cubism-05.jpg",
    "Cubism Style 3": "s_cubism-08.jpg",

    "Impressionist Style 1": "s_impressionist-02.jpg",
    "Impressionist Style 2": "s_impressionist-07.jpg",
    "Impressionist Style 3": "s_impressionist-11.jpg"
    
}

selected_image =st.sidebar.selectbox("Select the image which will be used as a reference",list(image_name_mapping.keys()) )

if st.sidebar.button('Search in Embedding Space'):
    try: 
                       
            def search_by_style(image_style_embeddings, images, selected_image, max_results=10):
                reference_image = image_name_mapping[selected_image]
                v0 = image_style_embeddings[reference_image]
                distances = {}
                for k,v in image_style_embeddings.items():
                  d = sc.spatial.distance.cosine(v0, v)
                  distances[k] = d

                sorted_neighbors = sorted(distances.items(), key=lambda x: x[1], reverse=False)

                st.title("Image Search by Style")

                f, ax = plt.subplots(1, max_results, figsize=(20, 10))
                for i, img in enumerate(sorted_neighbors[:max_results]):
                      ax[i].imshow(images[img[0]])
                      ax[i].set_axis_off()

                st.pyplot(f,use_container_width=True)
                plt.show()
    

               # images mostly match the reference style, although not perfectly
            search_by_style(image_style_embeddings, images, selected_image)
            
    except Exception as e:
        st.error(f"Error executing the query: {str(e)}")
