
from tensorflow.keras.models import load_model

import numpy as np


import xml.etree.ElementTree as ET
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import cv2


def get_files():
    with open('test_dataset1.csv', 'w', newline='') as csvfile:
        fieldnames = ['filename', 'x_min', 'y_min', 'x_max', 'y_max']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for dir in (['test']):
            for file in os.listdir(f'C:\\Users\\Komputer\\Desktop\\tablice\\dataset\\{dir}'):
                if file.startswith('Cars') and file.endswith('.xml'):
                    tree = ET.parse(f'C:\\Users\\Komputer\\Desktop\\tablice\\dataset\\{dir}\\{file}')
                    root = tree.getroot()
                    if not check_duplicate_plates('bndbox', root):
                        pass
                    else:
                        x_min, y_min, x_max, y_max = (extract_plate_postion_cars(root))
                        writer.writerow({'filename': file, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max})


def check_duplicate_plates(tagname, root):
    k = 0
    for child in root[4]:
        if child.tag == tagname:
            k = k + 1
    if k > 1:
        return False
    return True


def extract_plate_postion_cars(root):
    width = int(root[2][0].text)
    height = int(root[2][1].text)

    xmin = int(root[4][5][0].text) / width
    ymin = int(root[4][5][1].text) / height
    xmax = int(root[4][5][2].text) / width
    ymax = int(root[4][5][3].text) / height

    return xmin, ymin, xmax, ymax


WIDTH = 224
HEIGHT = 224


def show_img(index):
    print("pyramid\\images\\" + train_df["filename"].iloc[index])
    image = cv2.imread(f'./images/{train_df["filename"].iloc[index]}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(WIDTH, HEIGHT))

    tx = int(train_df["x_max"].iloc[index] * WIDTH)
    ty = int(train_df["y_max"].iloc[index] * HEIGHT)
    bx = int(train_df["x_min"].iloc[index] * WIDTH)
    by = int(train_df["y_min"].iloc[index] * HEIGHT)

    image = cv2.rectangle(image, (tx, ty), (bx, by), (0, 0, 255), 1)
    plt.imshow(image)
    plt.show()

#get_files()
train_df = pd.read_csv('train_dataset1.csv')
##train_df['filename'] = train_df['filename'].str.replace(r'.xml', '.png')

#load model
trained_model = load_model("my_model")

test_df = pd.read_csv('test_dataset1.csv')
test_df['filename'] = test_df['filename'].str.replace(r'.xml', '.png')



#to test uncomment this
#test_datagen = ImageDataGenerator(rescale=1./255)
#test_generator = test_datagen.flow_from_dataframe(
    #test_df,
    #directory='images',
    #x_col="filename",
    #y_col=["x_max", "y_max", "x_min", "y_min"],
    #target_size=(WIDTH, HEIGHT),
    #class_mode="raw",
    #batch_size=1)
#test_history = trained_model.evaluate(test_generator)
# check sample prediciton

img = cv2.resize(cv2.imread("Cars412.png") / 255.0, dsize=(224, 224))
y_hat = trained_model.predict(img.reshape(1, WIDTH, HEIGHT, 3)).reshape(-1) * 224

x_max, y_max = y_hat[0], y_hat[1]
x_min, y_min = y_hat[2], y_hat[3]

img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
image = cv2.rectangle(img, (x_max, y_max), (x_min, y_min), (0, 0, 255), 1)
plt.imshow(image)
plt.show()

