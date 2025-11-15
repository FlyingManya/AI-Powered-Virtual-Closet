import sqlite3
import os
import cv2
from sklearn.cluster import KMeans
from collections import Counter

def extract_color(img_path, k=1):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(img)
    label_counts = Counter(labels)
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    return dominant_color

# Create a connection to the database
conn = sqlite3.connect('fashion_recommender.db')
c = conn.cursor()

# Create a table for storing product metadata
c.execute('''CREATE TABLE IF NOT EXISTS products
             (id INTEGER PRIMARY KEY, 
              filename TEXT, 
              category TEXT, 
              color TEXT)''')

# Traverse the images directory and insert product metadata
image_dir = 'images'
for category in os.listdir(image_dir):
    category_path = os.path.join(image_dir, category)
    if os.path.isdir(category_path):
        for file in os.listdir(category_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(category_path, file)
                color = extract_color(file_path)
                color_str = f'{int(color[0])},{int(color[1])},{int(color[2])}'
                c.execute('INSERT INTO products (filename, category, color) VALUES (?,?,?)', 
                          (file_path, category, color_str))

conn.commit()
conn.close()
