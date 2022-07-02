import numpy as np
import cv2
import os

def crop_img(img_files, points):

    img = cv2.imread(img_files)
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    #method 1 smooth region
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    #method 2 not so smooth region
    # cv2.fillPoly(mask, points, (255))
    res = cv2.bitwise_and(img,img,mask = mask)
    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return cropped