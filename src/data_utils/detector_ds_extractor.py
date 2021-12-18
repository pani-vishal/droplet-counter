import sys
import os
import os.path as osp

sys.path.append("../..")
from src.data_utils.droplets_image_extractor import *
from src.utility import *
from tqdm import tqdm

### Functions ###
def extract_data(image_names, path_ip, path_op_img, path_op_lbl):
    for image_name in tqdm(image_names):
        # Create mask
        image_path = os.path.join(path_ip, image_name)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        circles = get_hough_circles(gray)

        # mask = np.zeros_like(gray, np.uint8)

        # for (x, y, r) in circles:
        #     cv2.circle(mask, (x, y), int(r * 1.1), (255, 255, 255), -1)

        # xs, ys = np.where(mask == 255)

        # gray = np.full_like(gray, np.mean(gray), np.uint8)
        # final_img[xs, ys] = gray[xs, ys]

        # Save image
        cv2.imwrite(osp.join(path_op_img, image_name), gray)

        # Save lbl
        lines = circles.copy()
        lines = np.hstack((lines, lines[:,-1][:, None])).astype(float)
        lines[:, [0, 2]] /= gray.shape[1]
        lines[:, [1, 3]] /= gray.shape[0]
        lines[:, [2, 3]] *= 2
        lines = lines.tolist()

        with open(osp.join(path_op_lbl, image_name[:-3]+"txt"), 'w') as f:
            for l in lines:
                line = f"0 {l[0]} {l[1]} {l[2]} {l[3]} \n"
                f.write(line)


path_ds = "../../datasets/"
path_ds_train = osp.join(path_ds, "droplets", "train", "original")
path_ds_test = osp.join(path_ds, "droplets", "test", "original")
path_op_train_img = osp.join(path_ds, "droplets", "detector_ds", "images", "train")
path_op_test_img = osp.join(path_ds, "droplets", "detector_ds", "images", "test")
path_op_train_lbl = osp.join(path_ds, "droplets", "detector_ds", "labels", "train")
path_op_test_lbl = osp.join(path_ds, "droplets", "detector_ds", "labels", "test")

list_train_imgs = os.listdir(path_ds_train)
list_test_imgs = os.listdir(path_ds_test)

extract_data(list_train_imgs, path_ds_train, path_op_train_img, path_op_train_lbl)
extract_data(list_test_imgs, path_ds_test, path_op_test_img, path_op_test_lbl)

