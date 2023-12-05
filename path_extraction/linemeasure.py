import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import math

vid = cv2.VideoCapture('IMG_1670.MOV') 

while vid.isOpened(): 
    ret, image = vid.read()
    y = image.shape[0]
    x = image.shape[1]
    halfx = int(x/2)
    yless = y - 1
    result = []

    
    whiteboard = np.ones((y, x, 3), np.uint8) * 256 
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lower_value_threshold = 50
    upper_value_threshold = 100
    lower_hue_threshold = 10
    upper_hue_threshold = 90

    # binary mask with value thresholds
    value_mask = cv2.inRange(v, lower_value_threshold, upper_value_threshold)

    # binary mask with huge thresholds
    hue_mask = cv2.inRange(h, lower_hue_threshold, upper_hue_threshold)

    # combine them
    composite_mask = cv2.bitwise_and(value_mask, hue_mask)

    segmented_image = cv2.bitwise_and(image, image, mask=composite_mask) 
    img_gry = cv2.cvtColor(segmented_image,cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(img_gry)

    img_blur = cv2.GaussianBlur(equalized, (5,5), 0) 

    ret, img_Otsubin = cv2.threshold(img_blur,10,255,cv2.THRESH_BINARY)
    
    imagetocanny = img_Otsubin

    edges = cv2.Canny(image=imagetocanny, threshold1=200, threshold2=200) # detect the canny edges

    # apply hough lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/100, 65, np.array([]),
                        100, 30)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.line(whiteboard,(x1,y1),(x2,y2),(255,0,0),2)



    # LEFT LINE
    y1 = y - 1
    x1 = halfx
    while x1 > 0:
        pixel_color = whiteboard[y1, x1]

        if all(pixel_color == [255, 0, 0]): 
            break
        y1 -= 1
        x1 -= 1
    cv2.line(image,(x1,y1),(halfx,y),(0,255,0),5)
    distance1 = int(math.hypot(halfx - x1, yless - y1))/100
    text1 = "Left Distance:" + str(distance1)
    result.append(int(math.hypot(halfx - x1, yless - y1)))
    cv2.putText(image, text1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # CENTRE LINE
    y2 = yless
    x2 = halfx
    while y2 > 0:
        pixel_color = whiteboard[y2, x2]

        if all(pixel_color == [255, 0, 0]):

            break
        y2 -= 1
    cv2.line(image,(halfx,y2),(halfx,y),(0,255,0),5)
    distance2 = int(math.hypot(halfx - x2, yless - y2))/50
    text2 = "Forward Distance:" + str(distance2)
    result.append(int(math.hypot(halfx - x2, yless - y2)))
    cv2.putText(image, text2, (halfx - 75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # RIGHT LINE
    y3 = yless
    x3 = halfx
    while x3 < x:
        pixel_color = whiteboard[y3, x3] 

        if all(pixel_color == [255, 0, 0]):
            break
        y3 -= 1
        x3 += 1
    cv2.line(image,(x3,y3),(halfx,y),(0,255,0),5)
    distance3 = int(math.hypot(halfx - x3, yless - y3))/100
    text3 = "Right Distance:" + str(distance3)
    result.append(int(math.hypot(halfx - x3, yless - y3)))
    cv2.putText(image, text3, (x-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    print(result)
    # resulting frame 
    cv2.imshow('frame', image) 
      
    if cv2.waitKey(1) == ord('q'): 
        break
  
vid.release()
cv2.destroyAllWindows()