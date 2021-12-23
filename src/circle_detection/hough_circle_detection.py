import cv2
import os
from pathlib import Path
import glob

def hough_circle_detection(images_path, maxRadius = 300, minRadius=5, minDist=10, param1=300, param2=0.9, dp=1.5):


    p = str(Path(images_path).resolve())
    if os.path.isdir(p):
        images_paths = sorted(glob.glob(p, recursive=True))
    elif os.path.isfile(p):
        images_paths = [p]
    else:
        raise Exception(f"ERROR: {p} does not exists")


    images = [cv2.imread(image_path) for image_path in images_paths]
    gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    list_circles = []
    for gray_image in gray_images:
        circles = cv2.HoughCircles(image=gray_image,
                                   method=cv2.HOUGH_GRADIENT_ALT,
                                   maxRadius=maxRadius,
                                   minRadius=minRadius,
                                   minDist=minDist,
                                   param1=param1,
                                   param2=param2,
                                   dp=dp
                                   )
        list_circles.append(circles[0])
    print(list_circles)
    return list_circles


import cv2
import numpy as np
import matplotlib.pyplot as plt


def hough_circle_detection_simple(image, hough_args=None):
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




class hough_circle_detection_gui_config:
    from PyQt5 import QtGui, QtCore, QtWidgets

    def __init__(self):
        self.config = {"maxRadius": "300",
                       "minRadius": "5",
                       "minDist": "10",
                       "param1": "300",
                       "param2": "0.9",
                       "dp": "1.5"
                       }

        self.form_widget = self.QtWidgets.QWidget()
        self.form_layout = self.QtWidgets.QFormLayout(self.form_widget)

        for key in self.config:
            input_line = self.QtWidgets.QLineEdit(self.config[key])
            input_line.textChanged.connect(lambda value: self.update_config(value, key))
            self.form_layout.addRow(key, input_line)

    def update_config(self, value, config_name):
        self.config[config_name] = value

    def get_config(self):
        return self.config


if __name__ == "__main__":
    image = cv2.imread('datasets/droplets/whole/original/3 hrs lambda 10 B0000000000.jpg')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    result = hough_circle_detection(gray)

    print(result)
