import math
import numpy as np
import tensorflow as tf
import cv2 as cv
import os
from feature_extraction import *

IMG_WIDTH = 64
IMG_HEIGHT = 32

class eqn:
    def __init__(self, y):
        self.y = y
    
    def eval(self, x):
        return (self.y[0] ** x) + (self.y[1] * x) + self.y[2]

def flatten(x):
    final = []
    for pos1 in range(len(x)):
        for pos2 in range(len(x[0])):
            for pos3 in range(len(x[0][0])):
                final.append(x[pos1][pos2][pos3])
    return final

features = []
for path in os.listdir(os.getcwd() + "\\Sidewalks"):
    img = cv.imread(os.getcwd() + "\\Sidewalks\\" + path)
    features.append(flatten([edge_detection(img, IMG_WIDTH, IMG_HEIGHT), hue_detection(img, IMG_WIDTH, IMG_HEIGHT)]))

train = np.array(features)

def custom_loss(features, lines):
    losses = []
    for pos in range(0, len(features)):
        x = features.numpy()[pos]
        y = lines.numpy()[pos]
        
        SD = 10
        HEIGHT = 40
        normal = lambda SD, HEIGHT, M, x: (1/(SD*math.sqrt(2*math.pi))) * ((math.e ** ((-1/2) * (((x-M) / SD) ** 2)))) * HEIGHT

        line = eqn(y)
        mid = int(len(x) / 2)

        max = 0
        loss1 = 0
        loss2 = 0
        for x_pos in range(0, IMG_WIDTH):
            y_line_pos = line.eval(x_pos)
            for y_pos in range(0, IMG_HEIGHT):
                val = normal(SD, HEIGHT, y_line_pos, y_pos)
                pos = int((IMG_WIDTH * y_pos) + x_pos)
                loss1 += x[pos] * val / 255
                loss2 += x[pos + mid] * val / 255
                max += val
        if max != 0:
            losses.append(min(((max - loss1) / max), ((max - loss2) / max)))
        else:
            losses.append(1)
    return tf.convert_to_tensor(losses)

def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dense(5, kernel_initializer='RandomNormal', activation='relu'),
    tf.keras.layers.Dense(3)
    ])

model = create_model()
model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'], run_eagerly=True)
model.fit(x=train, 
          y=train, 
          epochs=20, 
          validation_data=(train, train)
)