# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:12:25 2020

@author: Administrator
"""
import os
import shutil
import sys
import time
import cv2
import numpy as np
import time
import requests
import json
import sys
sys.path.append('/home/python-project/ocr/text_detection_ctpn')
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])

def ctpn(image_url):
    '''
    调用ctpn接口进行图片解析
    :param image_url:
    :return img: 图片矩阵
    :return boxes: 矩形框坐标
    :return scores: 置信度
    '''
    start = time.time()
    images = requests.get(image_url).content
    buf=np.asarray(bytearray(images),dtype="uint8")
    im=cv2.imdecode(buf,cv2.IMREAD_COLOR)
        
    img, (rh, rw) = resize_image(im)
    h, w, c = img.shape
    im_info = np.array([h, w, c]).reshape([1, 3])
    input_image = [img.tolist()]
    url = 'http://127.0.0.1:7501/v1/models/ctpn:predict'
    data = json.dumps({"name": 'tfserving-ctpn',"signature_name":'predict_images',"inputs":input_image})
    result = requests.post(url,data=data).json()
    cls_prob_val = np.asarray(result['outputs']['cls_prob_output'], dtype = np.float32)
    bbox_pred_val = np.asarray(result['outputs']['bbox_pred_output'], dtype = np.float32)
    textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)

    scores = textsegs[:, 0]
    textsegs = textsegs[:, 1:5]

    textdetector = TextDetector(DETECT_MODE='H')
    boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
    boxes = np.array(boxes, dtype=np.int)
    cost_time = (time.time() - start)
    print("cost time: {:.2f}s".format(cost_time))
    return img, boxes, scores

if __name__ == '__main__':
    url = 'https://i.loli.net/2020/03/25/a8yglr6i5USwmeW.jpg'
    result = ctpn(url)
    print(result)