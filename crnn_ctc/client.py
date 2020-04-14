# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:42:22 2020

@author: JianjinL
"""

import cv2
import requests
import json
import numpy as np

# 图片大小
OUTPUT_SHAPE = (256,32,3)

def text_ocr(img):
    '''
    矩形框文字识别
    '''
    image = cv2.resize(img, (OUTPUT_SHAPE[0], OUTPUT_SHAPE[1]), 3)
    image = image.reshape(OUTPUT_SHAPE)
    input_image = [image.tolist()]
    seq_len = [15]
    url = 'http://127.0.0.1:6501/v1/models/crnn_ctc:predict'
    data = json.dumps({
            "name": 'tfserving-ocr',
            "signature_name": 'predict_images',
            "inputs":{
                    "inputs": input_image,
                    "seq_len": seq_len}})
    text = requests.post(url,data=data).json()
    if 'outputs' in text:
        result = ''.join([answer[str(char)] for char in text['outputs'][0]])
    return result

if __name__ == '__main__':
    #映射结果
    with open('../dataset/create_data/english/answer.json', 'r') as f:
        answer = json.load(f)
    url = 'https://i.loli.net/2020/03/31/ZimX6ufSL1KxQw9.jpg'
    images = requests.get(url).content
    buf=np.asarray(bytearray(images),dtype="uint8")
    im=cv2.imdecode(buf,cv2.IMREAD_COLOR)
    result = text_ocr(im)
    print(result)
else:
    #映射结果
    with open('./dataset/create_data/english/answer.json', 'r') as f:
        answer = json.load(f)