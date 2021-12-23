import sys
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn

sys.path.append("yolov5")

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


def yolo5_circle_detection(path_to_images,
                       weights="models/yolo_best.pt",
                       imgsz=(1280, 1280),  # inference size (height, width)
                       conf_thres=0.60,  # confidence threshold
                       iou_thres=0.45,  # NMS IOU threshold
                       max_det=1000,  # maximum detections per image
                       device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                       classes=None,  # filter by class: --class 0, or --class 0 2 3
                       agnostic_nms=False,  # class-agnostic NMS
                       augment=False,  # augmented inference
                       half=False,  # use FP16 half-precision inference
                       dnn=False,  # use OpenCV DNN for ONNX inference
                       ):
    max_det = int(max_det)
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    dataset = LoadImages(path_to_images, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    list_circles = []
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=augment, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        det = pred[0]

        for i, det in enumerate(pred):
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            circles = torch.empty((det.shape[0], 3))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                circles[:, 0] = (det[:, 0] + det[:, 2]) / 2
                circles[:, 1] = (det[:, 1] + det[:, 3]) / 2
                circles[:, 2] = (det[:, 3] - det[:, 1] + det[:, 2] - det[:, 0]) / 4

            list_circles.append(circles)

    return list_circles

class yolo5_circle_detection_gui_config:
    from PyQt5 import QtGui, QtCore, QtWidgets

    def __init__(self):
        self.config = {"weights": '"models/yolo_best.pt"',
                    "iou_thres": "0.45",
                    "conf_thres": "0.60",
                    "max_det": "1000"
                    }

        self.form_widget = self.QtWidgets.QWidget()
        self.form_layout = self.QtWidgets.QFormLayout(self.form_widget)

        for key in self.config:
            input_line = self.QtWidgets.QLineEdit(self.config[key])
            input_line.textChanged.connect(lambda value, key=key: self.update_config(value, key))
            self.form_layout.addRow(key, input_line)

    def update_config(self, value, config_name):
        print(value, config_name)
        self.config[config_name] = value

    def get_config(self):
        return self.config
