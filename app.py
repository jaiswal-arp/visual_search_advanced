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

if st.sidebar.button('Generate'):
    # Compute and Visualize the Style Embeddings using inbuilt models
    def load_image(image):
      image = plt.imread(image)
      img = tf.image.convert_image_dtype(image, tf.float32)
      img = tf.image.resize(img, [400, 400])
      img = img[tf.newaxis, :] # shape -> (batch_size, h, w, d)
      return img

#
# content layers describe the image subject
#
    content_layers = ['block5_conv2'] 

#
# style layers describe the image style
# we exclude the upper level layes to focus on small-size style details
#
    style_layers = [ 
        'block1_conv1',
        'block2_conv1',
        'block3_conv1', 
        #'block4_conv1', 
        #'block5_conv1'
    ] 

    def selected_layers_model(layer_names, baseline_model):
      outputs = [baseline_model.get_layer(name).output for name in layer_names]
      model = Model([vgg.input], outputs)
      return model

# style embedding is computed as concatenation of gram matrices of the style layers
    def gram_matrix(input_tensor):
      result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
      input_shape = tf.shape(input_tensor)
      num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
      return result/(num_locations)

    class StyleModel(tf.keras.models.Model):
        def __init__(self, style_layers, content_layers):
          super(StyleModel, self).__init__()
          self.vgg =  selected_layers_model(style_layers + content_layers, vgg)
          self.style_layers = style_layers
          self.content_layers = content_layers
          self.num_style_layers = len(style_layers)
          self.vgg.trainable = False

        def call(self, inputs):
          # scale back the pixel values
          inputs = inputs*255.0
          # preprocess them with respect to VGG19 stats
          preprocessed_input = preprocess_input(inputs)
          # pass through the reduced network
          outputs = self.vgg(preprocessed_input)
          # segregate the style and content representations
          style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

          # calculate the gram matrix for each layer
          style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

          # assign the content representation and gram matrix in
          # a layer by layer fashion in dicts
          content_dict = {content_name:value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

          style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

          return {'content':content_dict, 'style':style_dict}

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    def image_to_style(image_tensor):
      extractor = StyleModel(style_layers, content_layers)
      return extractor(image_tensor)['style']

    def style_to_vec(style):
      # concatenate gram matrics in a flat vector
      return np.hstack([np.ravel(s) for s in style.values()]) 

#
# Print shapes of the style layers and embeddings
# 
    
    image_tensor = load_image(image_paths[0])
    style_tensors = image_to_style(image_tensor)
    for k,v in style_tensors.items():
        print(f'Style tensor {k}: {v.shape}')
    style_embedding = style_to_vec( style_tensors )
    print(f'Style embedding: {style_embedding.shape}')

#
# compute styles
#
    image_style_embeddings = {}
    for image_path in tqdm(image_paths): 
        image_tensor = load_image(image_path)
        style = style_to_vec( image_to_style(image_tensor) )
        image_style_embeddings[ntpath.basename(image_path)] = style



#
# Visualize the 2D-projection of the embedding space with example images (thumbnails)
#
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
    
st.sidebar.write("")
