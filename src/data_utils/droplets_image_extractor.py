import cv2
import numpy as np
import math

def droplets_image_extractor(image, circle_detection_method, circle_detection_args = None, remove_background = True, border_circles_included = True):

    circles = circle_detection_method(image, circle_detection_args)

    height, width = image.shape[:2]

    result = []

    for (x, y, r) in circles[0, :]:
        if not border_circles_included and (x - r < 0 or x + r > width or y - r < 0 or y + r > height):
            continue

        x_min = math.floor(max(x - r, 0))
        x_max = math.ceil(min(x + r, width))
        y_min = math.floor(max(y - r, 0))
        y_max = math.ceil(min(y + r, height))

        droplet = image[y_min:y_max, x_min:x_max]

        if remove_background:
            mask = np.zeros_like(droplet)
            mask = cv2.circle(mask, (round(r), round(r)), round(r), (255,255,255), -1)
            masked_droplet = cv2.bitwise_and(droplet, mask)
            result.append(masked_droplet)
        else:
            result.append(droplet)

    return result
    
