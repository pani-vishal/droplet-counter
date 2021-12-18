### Script for helper functions ###
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def get_hough_circles(image_gray, maxRadius=200, minRadius=10, dp=1.5, param1=300, param2=0.9):
    """ Returns the detected circles rather than saving them """
    
    circles = cv2.HoughCircles(image=image_gray,
                               method=cv2.HOUGH_GRADIENT_ALT,
                               dp=dp,
                               minDist=2*minRadius,
                               param1=param1,
                               param2=param2,
                               minRadius=minRadius,
                               maxRadius=maxRadius
                              )
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        if "4.5.3" in cv2.__version__:
            circlesRound = np.round(circles[:,0,:]).astype("int")
        elif "4.5.4" in cv2.__version__:
            circlesRound = np.round(circles[0,:]).astype("int")
        else:
            print("Incompatible opencv version, please use either 4.5.3 or 4.5.4")
            exit()
        #loop over the (x, y) coordinates and radius of the circles
        return circlesRound
    else:
        return np.array([])