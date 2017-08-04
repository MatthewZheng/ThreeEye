#! usr/bin/env python2.7
_author_ = "Matthew Zheng"
_purpose_ = "Test detection methods for changes in the model"

#imports
import sys
import cv2
import numpy as nmp

def main():
    #setup video capture on webcame
    vid = cv2.VideoCapture(1)

    #Initialize bg/fg segmentation algorithim
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
    fgBg = cv2.createBackgroundSubtractorMOG2()

    #loop for continuous video
    while True:
        #read all frames from video cam
        ret, frame = vid.read()
        fgMask = fgBg.apply(frame)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

        cv2.imshow('frame', fgMask)

        #exit script with 'e'
        if(cv2.waitKey(1) & 0xFF == ord('e')):
            break


    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
