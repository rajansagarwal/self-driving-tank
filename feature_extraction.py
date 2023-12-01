import cv2 as cv
import os


IMG_WIDTH = 64
IMG_HEIGHT = 32

def edge_detection(frame, width, height): 
    frame = cv.resize(frame, (width, height))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.medianBlur(gray,11)
    edges = cv.Canny(blurred, 40, 100)
    return edges

def hue_detection(frame, width, height):
    frame = cv.resize(frame, (width, height))
    hsl_img = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
    for pos1 in range(len(hsl_img)):
        for pos2 in range(len(hsl_img[0])):
            hsl_img[pos1][pos2][1] = 100
            hsl_img[pos1][pos2][2] = 50
    blurred = cv.medianBlur(hsl_img,11)
    edges = cv.Canny(blurred, 40, 100)
    return edges


# img = cv.imread(os.getcwd() + "\\Sidewalks\\P10.jpg")


# cv.imshow("Display window", edge_detection(img, IMG_WIDTH, IMG_HEIGHT))
# k = cv.waitKey(0)

# cv.imshow('HLS Image', hue_detection(img, IMG_WIDTH, IMG_HEIGHT))
# k = cv.waitKey(0)