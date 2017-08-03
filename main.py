#! usr/bin/env python2.7
_author_ = "Matthew Zheng"
_purpose_ = "OpenCV 3D model error detection"

#imports
import sys
import cv2
import numpy as nmp
from matplotlib import pyplot as pplot

def main():
    #setup video capture on webcame
    vid = cv2.VideoCapture(1)

    #loop for continuous video
    while True:
        #read all frames from video cam
        ret, frame = vid.read()

        #convert to gray and find harris corners
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = nmp.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        frame[dst>0.01*dst.max()]=[0,255,0]

        #Apply thresholding value to remove unimportant features
        ret, thres = cv2.threshold(frame,127,255,cv2.THRESH_BINARY)

        #convert frames to HSV
        hsv = cv2.cvtColor(thres, cv2.COLOR_BGR2HSV)

        #Show result
        cv2.imshow('Video', hsv)
        cv2.imshow('dst', frame)

        #exit script with 'e'
        if(cv2.waitKey(1) & 0xFF == ord('e')):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
