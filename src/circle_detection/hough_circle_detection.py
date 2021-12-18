import cv2
import numpy as np
import matplotlib.pyplot as plt


def hough_circle_detection(image, hough_args=None):
    hough_args_default = {"maxRadius": 300,
                          "minRadius": 5,
                          "minDist": 10,
                          "param1": 300,
                          "param2": 0.9,
                          "dp": 1.5,
                          "method": cv2.HOUGH_GRADIENT_ALT
                          }

    if hough_args is None:
        hough_args = hough_args_default
    else:
        hough_args = hough_args_default.update(hough_args)

    circles = cv2.HoughCircles(image=image,
                               **hough_args
                               )

    return circles


if __name__ == "__main__":
    image = cv2.imread('datasets/droplets/whole/original/3 hrs lambda 10 B0000000000.jpg')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    result = hough_circle_detection(gray)

    print(result)
