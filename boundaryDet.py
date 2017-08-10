#! usr/bin/env python2.7
_author_ = "Matthew Zheng"
_purpose_ = "Identify the boundaries on an object"

#imports
import sys
import cv2
import imutils
import numpy as nmp
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse

def midpoint(pointA, pointB):
    return((pointA[0] + pointB[0]) * 0.5, (pointA[1] + pointB[1]) * 0.5)
def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help="add path to image")
    ap.add_argument("-w", "--width", required = True, type=float, help="width of known object (mm).")
    return(vars(ap.parse_args()))

def main():
    #parse incoming
    args = parser()

    #setup cam+settings (for image)
    img = cv2.imread(args["image"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    #edge detection
    edges = cv2.Canny(gray, 50, 100)
    edges = cv2.dilate(edges, None, iterations = 1)
    edges = cv2.erode(edges, None, iterations = 1)

    #find contours
    cntrs = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if imutils.is_cv2() else cntrs[1]

    #sort contours
    (cntrs, _) = contours.sort_contours(cntrs)
    pPerM = None

    for num in cntrs:
        if cv2.contourArea(num) < 50:
            continue

        original = img.copy()
        #compute bounding box
        box = cv2.minAreaRect(num)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = nmp.array(box, dtype="int")

        #order points
        box = perspective.order_points(box)
        cv2.drawContours(original, [box.astype("int")], -1, (0, 255, 0), 2)

        #loop over and draw points
        for (x,y) in box:
            cv2.circle(original, (int(x), int(y)), 5, (0, 0, 255) , -1)

        (topL, topR, botL, botR) = box
        #find midpoints
        (topLtopRX, topLtopRY) = midpoint(topL, topR)
        (botLbotRX, botLbotRY) = midpoint(botL, botR)
        (topLbotLX, topLbotLY) = midpoint(topL, botL)
        (topRbotRX, topRbotRY) = midpoint(topR, botR)

        #draw out
        cv2.circle(original, (int(topLtopRX), int(topLtopRY)), 5, (255, 0, 0), -1)
        cv2.circle(original, (int(botLbotRX), int(botLbotRY)), 5, (255, 0, 0), -1)
        cv2.circle(original, (int(topLbotLX), int(topLbotLY)), 5, (255, 0, 0), -1)
        cv2.circle(original, (int(topRbotRX), int(topRbotRY)), 5, (255, 0, 0), -1)

        #draw out lines between midpoints
        cv2.line(original, (int(topLtopRX), int(topLtopRY)), (int(botLbotRX), int(botLbotRY)),
		(255, 0, 255), 2)
        cv2.line(original, (int(topLbotLX), int(topLbotLY)), (int(topRbotRX), int(topRbotRY)),
		(255, 0, 255), 2)

        dHeightE = dist.euclidean((topLtopRX, topLtopRY), (botLbotRX, botLbotRY))
        dWidthE = dist.euclidean((topLbotLX, topLbotLY), (topRbotRX, topRbotRY))

        if pPerM == None:
            pPerM = dWidthE/args["width"]

        #compute size
        distHeight = dHeightE / pPerM
        distWidth = dWidthE / pPerM

        #draw labels on image
        cv2.putText(original, "{:.2f}mm".format(distHeight), (int(topLtopRX - 15), int(topLtopRY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255) , 2)
        cv2.putText(original, "{:.2f}mm".format(distWidth), (int(topRbotRX - 15), int(topRbotRY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255) , 2)

        cv2.imshow("Image", original)
        #exit script with 'e'
        cv2.waitKey(0)



if __name__ == "__main__":
    main()
