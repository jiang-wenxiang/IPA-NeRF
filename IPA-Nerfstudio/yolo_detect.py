import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Resize
from ultralytics import RTDETR, NAS, YOLO


def yolo_result_img_save(image, results, save_dir, save_path, yolo_model_name):
    numpy_img = image.copy()
    log_str = ""
    log_str += "[" + save_dir + yolo_model_name + "_" + save_path + "]\n"

    for rect_i in range(len(results[0].boxes.cls)):
        rect = results[0].boxes.xyxy[rect_i]
        cls_name = results[0].names[int(results[0].boxes.cls[rect_i])] + " " + str(
            float(results[0].boxes.conf[rect_i]) * 100)[:5] + "%"
        log_str += cls_name + "\n"

        # numpy_img_size = numpy_img.shape
        # rect = rect / 640
        # rect = torch.clip(rect, 0, 1)
        # numpy_img_h = numpy_img_size[0]
        # numpy_img_w = numpy_img_size[1]
        # rect[0] = rect[0] * numpy_img_w
        # rect[1] = rect[1] * numpy_img_h
        # rect[2] = rect[2] * numpy_img_w
        # rect[3] = rect[3] * numpy_img_h

        cv2.rectangle(numpy_img, (int(rect[0]), int(rect[1])),
                      (int(rect[2]), int(rect[3])), (0, 0, 255),
                      thickness=2)
        cv2.rectangle(numpy_img, (int(rect[2] - 400), int(rect[1] + 50)),
                      (int(rect[2]), int(rect[1])), (0, 0, 255),
                      thickness=-1)
        cv2.putText(numpy_img, cls_name, (int(rect[2] - 380), int(rect[1] + 30)),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    plt.imsave(save_dir + yolo_model_name + "_" + save_path, numpy_img)
    plt.imsave(save_dir + yolo_model_name + "_" + save_path, numpy_img)
    return log_str


device = "cuda:0"

model_name_list = ["YOLOv5n", "YOLOv8n", "YOLOv8mw", "YOLOv9c", "RTDETR"]
# model_name = "YOLOv5n"
# model_name = "YOLOv8n"
# model_name = "YOLOv8mw"
# model_name = "YOLOv9c"
# model_name = "RTDETR"
log_str = ""

# log_name = "000149_0"
# log_name = "atbg"
# log_name = "000149_0"
# log_name = "bg"
# log_name = "bg"

# save_dir = "outputs/unnamed/nerfacto/bicycle/attack/"
# save_dir = "outputs/unnamed/nerfacto/bicycle/bg_and_atbg/"
# save_dir = "outputs/unnamed/nerfacto/road_2/attack/"
# save_dir = "outputs/unnamed/nerfacto/road_2/bg_and_atbg/"

scene_name = "kitchen"

save_dir_list = ["outputs/unnamed/nerfacto/"+scene_name+"/bg_and_atbg/",
                 "outputs/unnamed/nerfacto/"+scene_name+"/bg_and_atbg/",
                 "outputs/unnamed/nerfacto/"+scene_name+"/attack/"
                 ]

log_name_list = ["atbg",
                 "bg",
                 "000149_0"
                 ]

for i in range(len(save_dir_list)):
    log_name = log_name_list[i]
    save_dir = save_dir_list[i]
    save_path = log_name + ".jpg"
    for model_name in model_name_list:

        # init YOLO
        # Load a model
        model_load_dict = {
            "YOLOv5n": "yolov5nu.pt",
            "YOLOv8n": 'yolov8n.pt',
            "YOLOv9c": "yolov9c.pt",
            "YOLOv8mw": "yolov8m-world.pt",
            "RTDETR": "rtdetr-l.pt",

            "YOLOv3n": "yolov3n.pt",
            "YOLOv10n": "yolov10n.pt",
            "YOLOv5m": "yolov5m.pt",
            "YOLOv8m": 'yolov8m.pt',
            "YOLOv10m": "yolov10m.pt",
            "NAS": "yolo_nas_l.pt",
        }
        model_load_path = ""
        model_load_name = model_load_dict[model_name]
        if "yolo" in model_load_name:
            model = YOLO(model_load_path + model_load_name)
        elif "rtdetr" in model_load_name:
            model = RTDETR(model_load_path + model_load_name)
        else:
            model = NAS(model_load_path + model_load_name)
        model.to(device)

        image_path = os.path.join(save_dir, save_path)
        image_name = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)
        im_data = np.array(image.convert("RGB"))

        results = model(image_path)

        log_str += yolo_result_img_save(im_data, results, save_dir, save_path, model_name)

    with open(save_dir + log_name + "_log.txt", "w") as f:
        f.write(log_str)
        f.close()


