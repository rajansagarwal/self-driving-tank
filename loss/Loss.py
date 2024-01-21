import cv2 as cv
import os
import numpy
import math
from feature_extraction import *


IMG_WIDTH = 1280
IMG_HEIGHT = 780


# class eqn:
#     def __init__(self, y):
#         self.y = y
    
#     def eval(self, x):
#         try:
#             one = abs(self.y[0]) ** (x * self.y[1])
#             two = abs(self.y[2]) * (x ** self.y[3])
#             return ((self.y[0]/abs(self.y[0])) * one) + ((self.y[2]/abs(self.y[2])) * two) + self.y[4]
#         except ZeroDivisionError:
#             print("here")
#             one = abs(self.y[0]) ** (x * self.y[1])
#             two = abs(self.y[2]) * (x ** abs(self.y[3]))
#             return ((self.y[0]/abs(self.y[0])) * one) + ((self.y[2]/abs(self.y[2])) * two) + self.y[4]

class eqn:
    def __init__(self, y):
        self.y = y
    
    def eval(self, x):
        return (self.y[0] ** x) + (self.y[1] * x) + self.y[2]

def flatten(x):
    final = []
    for pos1 in range(len(x)):
        for pos2 in range(len(x[0])):
            final.append(x[pos1][pos2])
    return final

def expand(x):
    ls1 = []
    for pos1 in range(0, IMG_HEIGHT):
        row = []
        for pos2 in range(0, IMG_WIDTH):
            row.append(x[(pos1 * IMG_WIDTH) + pos2])
        ls1.append(row)
    return numpy.array(ls1)

img = cv.imread(os.getcwd() + "\\Sidewalks\\" + "P2.jpg")
cv.imshow('HLS Image', hue_detection(img, IMG_WIDTH, IMG_HEIGHT))
k = cv.waitKey(0)
feat1 = flatten(edge_detection(img, IMG_WIDTH, IMG_HEIGHT))
feat2 = flatten(hue_detection(img, IMG_WIDTH, IMG_HEIGHT))
if sum(feat1) < sum(feat2):
    x = feat2
else:
    x = feat1


def custom_loss(x, y):
    SD = 1
    HEIGHT = 5
    normal = lambda SD, HEIGHT, M, x: (1/(SD*math.sqrt(2*math.pi))) * ((math.e ** ((-1/2) * (((x-M) / SD) ** 2)))) * HEIGHT

    line = eqn(y)
    x = expand(x)

    temp = []
    for x_pos in range(0, IMG_WIDTH):
        y_line_pos = line.eval(x_pos)
        col = []
        for y_pos in range(0, IMG_HEIGHT):
            col.append(normal(SD, HEIGHT, y_line_pos, y_pos))
        temp.append(col)

    weights = []
    for x_pos in range(len(temp[0])):
        row = []
        for y_pos in range(len(temp)):
            row.append(temp[y_pos][x_pos])
        weights.append(row)
    cv.imshow('HLS Image', numpy.array(weights))
    k = cv.waitKey(0)
        
    loss = 0
    big = 0
    for pos1 in range(len(weights)):
        for pos2 in range(len(weights[0])):
            loss += weights[pos1][pos2] * x[pos1][pos2] / 255
            big += weights[pos1][pos2]
    return (big - loss) / big

def optimized_custom_loss(x, y):

    SD = 1
    HEIGHT = 5
    normal = lambda SD, HEIGHT, M, x: (1/(SD*math.sqrt(2*math.pi))) * ((math.e ** ((-1/2) * (((x-M) / SD) ** 2)))) * HEIGHT

    line = eqn(y)

    big = 0
    loss = 0
    for x_pos in range(0, IMG_WIDTH):
        y_line_pos = line.eval(x_pos)
        for y_pos in range(0, IMG_HEIGHT):
            val = normal(SD, HEIGHT, y_line_pos, y_pos)
            pos = int((IMG_WIDTH * y_pos) + x_pos)
            loss += x[pos] * val / 255
            big += val
    return (big - loss) / big

y = [1, -32, 119]
print(custom_loss(x, y))
print(optimized_custom_loss(x, y))