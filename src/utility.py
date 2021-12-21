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


def create_circular_mask(shape, center=None, radius=None):
    """ ref: https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array """

    h, w = shape

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def calculate_hist_sum(image, circle):
    x_1 = int(circle[0] - 0.75 * circle[2])
    y_1 = int(circle[1] - 0.75 * circle[2])
    x_2 = int(circle[0] + 0.75 * circle[2])
    y_2 = int(circle[1] + 0.75 * circle[2])
    droplet_image = image[y_1:y_2, x_1:x_2]
    histogram = cv2.calcHist([droplet_image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    hist_normed = cv2.normalize(histogram, histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist_normed, hist_normed.sum()