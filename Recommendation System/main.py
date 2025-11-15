import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import sqlite3

# Load data
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))
color_list = np.array(pickle.load(open('colors.pkl', 'rb')))

# Load model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def extract_color(img_path, k=1):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(img)
    label_counts = Counter(labels)
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    return dominant_color

def recommend(features, feature_list, color, color_list, filter_type):
    n_neighbors = min(len(feature_list), 100)  # Dynamically set n_neighbors
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])

    conn = sqlite3.connect('fashion_recommender.db')
    c = conn.cursor()

    filtered_indices = []
    for idx in indices[0]:
        filename = filenames[idx]
        c.execute('SELECT category, color FROM products WHERE filename=?', (filename,))
        result = c.fetchone()
        if result and filter_type in result[0]:
            product_color = np.array([int(c) for c in result[1].split(',')])
            if np.allclose(product_color, color, atol=50):  # Adjust tolerance as needed
                filtered_indices.append(idx)

    if len(filtered_indices) < 5:
        remaining_indices = [idx for idx in indices[0] if idx not in filtered_indices]
        filtered_indices += remaining_indices[:5 - len(filtered_indices)]

    return filtered_indices[:5]

uploaded_file = st.file_uploader("Choose an image")
filter_type = st.selectbox("Select a filter", ["T-Shirt", "Cap", "Jeans", "Earrings"])

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        color = extract_color(os.path.join("uploads", uploaded_file.name))

        indices = recommend(features, feature_list, color, color_list, filter_type)

        cols = st.columns(5)
        for i, col in enumerate(cols):
            if i < len(indices):
                with col:
                    st.image(filenames[indices[i]])
    else:
        st.header("Some error occurred in file upload")
