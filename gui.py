import argparse
import os
import os.path as osp
import cv2

from screeninfo import get_monitors

## Arguments
parser = argparse.ArgumentParser(description='Arguments for the GUI script')
parser.add_argument('-d', '--dir', type=str, default="./datasets/droplets/test/original", help="Path to the directory containing the images")
args = parser.parse_args()

## Generating the path list for all the images
path_dir = args.dir
list_images = os.listdir(path_dir)
list_path_images = [osp.join(path_dir, x) for x in list_images]

## Settings
monitor = get_monitors()[0]
width = monitor.width
height = monitor.height - 100

## Global vars
image = None
window_name = None

def on_change(value):
    global image, window_name
    image_copy = image.copy()
    cv2.putText(image_copy, str(value), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, color=(24,255,24, 10), thickness=5)
    # Display image
    cv2.imshow(window_name, image_copy)

# Video settings
idx = 0
while True:

    window_name = f"{list_images[idx]}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 0, 0)
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(window_name, width, height)
    image = cv2.imread(list_path_images[idx])

    cv2.createTrackbar('slider', window_name, 0, 100, on_change)
    cv2.imshow(window_name, image)
    key = cv2.waitKey(0)

    if key == ord("a"):
        if idx != 0:
            idx -= 1
        cv2.destroyWindow(window_name)
    if key == ord("d"):
        if idx < len(list_images)-1:
            idx += 1
        cv2.destroyWindow(window_name)
    if key == 27:
        break
cv2.destroyAllWindows()