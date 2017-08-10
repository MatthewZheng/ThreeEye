#! usr/bin/env python2.7
_author_ = "Matthew Zheng"
_purpose_ = "Takes a picture"

#imports
import cv2

def getImg():
    cam = cv2.VideoCapture(1)
    ret, img = cam.read()
    del(cam)
    return img

capture = getImg()
file = "C:\Users\Zhenger\Desktop\MLH\ThreeEye\cam-cap.png"
cv2.imwrite(file, capture)
