# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 17:44:59 2022

@author: user
"""

import streamlit as st
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import sklearn.metrics.pairwise as pt
from tensorflow.keras.preprocessing.image import img_to_array

from PIL import Image

model = ResNet50(include_top = False, weights = "imagenet", input_shape = (224,224,3))

def extract_features(image, model):
    expand_image = np.expand_dims(image, axis = 0)
    preprocess_image = preprocess_input(expand_image)
    feature_vector = model.predict(preprocess_image)
    flatten_feature = feature_vector.flatten()
    norm_feature = flatten_feature/np.linalg.norm(flatten_feature)
    return norm_feature

def similarity(image1, image2, model):
    image1 = image1.resize((224,224))
    image2 = image2.resize((224,224))
    image1 = img_to_array(image1)
    image2 = img_to_array(image2)
    feature1 = extract_features(image1, model)
    feature2 = extract_features(image2, model)
    cos_sim = 100 * (np.dot(feature1, feature2)/(np.linalg.norm(feature1)*np.linalg.norm(feature2)))
    if cos_sim > 70:
        st.write("Images are very similar")
    elif cos_sim > 50:
        st.write("Images are some what similar")
    elif cos_sim > 20:
        st.write("images have minor similarity")
    else:
        st.write("Images have no similarity")
        
st.title("Similarity checking application")
st.header("This application is used to check similarity between images, make sure images are discoverable in your Device")
data1 = st.file_uploader(label= "Please Upload your First Image",accept_multiple_files=False)
data2 = st.file_uploader(label= "Please Upload your Second Image", accept_multiple_files=False)
if data1 is not None and data2 is not None:
    npimg1 = Image.open(data1)
    npimg2 = Image.open(data2)
if st.button("Display Images"):
    with st.container():
        st.image(npimg1)
        st.image(npimg2)
if st.button("Check!"):
    similarity(npimg1, npimg2, model)