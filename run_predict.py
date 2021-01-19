import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
from core.config import cfg
import core.utils as utils
import argparse

def find_bbox(path, confidence_score=0.7, iou=0.5):
    bbox_list = []
    images = []
    model = tf.keras.models.load_model('SavedModel/YOLOv3_model', compile=False)

    INPUT_SIZE = cfg.TEST.INPUT_SIZE
    SCORE_THRESHOLD = confidence_score
    IOU_THRESHOLD = iou

    for filename in os.listdir(path):
        full_path_image = path + '/' + filename
        images.append(full_path_image)
    
    for path_to_image in images:
        image = cv2.imread(path_to_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_size = image.shape[:2]
        image_data = utils.image_preporcess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
        image_data = image_data[np.newaxis, ...].astype(np.float32) # (1, width, height, 3)

        pred_bbox = model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, SCORE_THRESHOLD)
        bboxes = utils.nms(bboxes, IOU_THRESHOLD, method='nms')
        bboxes.append(path_to_image)
        bbox_list.append(bboxes)
    
    return bbox_list
    # * 리턴은 다음과 같은 모양입니다.
    # * [array([5.73670227e+02, 2.99158386e+02, 1.00067535e+03, 8.91056885e+02, 8.61678839e-01, 1.00000000e+00]), './folder_path/image_15.png']