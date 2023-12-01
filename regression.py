import os
import cv2 as cv
from feature_extraction import *
import math

IMG_WIDTH = 64
IMG_HEIGHT = 32

A_TRY_DISTANCE = 0.05
B_TRY_DISTANCE = 1
C_TRY_DISTANCE = 5
COUNTER_MAX = 100



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

def loss(x, y):  
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

img = cv.imread(os.getcwd() + "\\Sidewalks\\" + "P10.jpg")
feat1 = flatten(edge_detection(img, IMG_WIDTH, IMG_HEIGHT))
feat2 = flatten(hue_detection(img, IMG_WIDTH, IMG_HEIGHT))
if sum(feat1) < sum(feat2):
    x = feat2
else:
    x = feat1

def left_45_dis(y):
    line = eqn(y)
    x = IMG_WIDTH / 2
    while (x > line.eval(x)):
        x -= 1
    return math.sqrt(2) * ((IMG_WIDTH / 2) - x)

def right_45_dis(y):
    line = eqn(y)
    x = IMG_WIDTH / 2
    while (x > line.eval(x)):
        x += 1
    return math.sqrt(2) * ((IMG_WIDTH / 2) - x)



def get_line(x):
    low = 1
    a = 10
    for b in range(-IMG_HEIGHT, IMG_HEIGHT, 1):
        for c in range(-IMG_HEIGHT * 4, IMG_HEIGHT * 4, 1):
            val = loss(x, [a/10, b, c])
            if val < low and val > 0:
                low = val
                best = [a/10, b, c]
                print(val, best)
                if val < 0.1:
                    return best
    return best

def rec_line(x, old_y, old_loss, counter):
    [a, b, c] = old_y
    a1 = a + A_TRY_DISTANCE
    ya1 = loss(x, [a1, b, c])
    if ya1 - old_loss < 0:
        new_a = a1
    elif a > 0:
        new_a = a - A_TRY_DISTANCE
    else:
        new_a = a
    
    b1 = b + B_TRY_DISTANCE
    yb1 = loss(x, [a, b1, c])
    if yb1 - old_loss < 0:
        new_b = b1
    else:
        new_b = b - B_TRY_DISTANCE
    

    c1 = c + C_TRY_DISTANCE
    yc1 = loss(x, [a, b, c1])
    if yc1 - old_loss < 0:
        new_c = c1
    else:
        new_c = c - C_TRY_DISTANCE
    
    if counter < COUNTER_MAX:
        l = loss(x, [new_a, new_b, new_c])
        print([new_a, new_b, new_c], l)
        return rec_line(x, [new_a, new_b, new_c], l, counter + 1)
    else:
        return [new_a, new_b, new_c]

# print(rec_line(x, [0.5, -10, 25], loss(x, [0.5, -10, 25]), 0))
# print(loss(x, [0, -5, 145]))
# print(loss(x, [1.17, 1.9, -90]))
print(get_line(x))
# print(left_45_dis([1, -3.9, 115]))