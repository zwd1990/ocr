# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:57:22 2020

@author: JainjinL
"""

from text_detection_ctpn.main.client import ctpn
from crnn_ctc.client import text_ocr

def ocr(url):
    '''
    图片文本识别
    '''
    result = {}
    img, boxes, scores = ctpn(url)
    #遍历每个矩形框
    for i in range(len(boxes)):
        crop_img = img[boxes[i][1]:boxes[i][5], boxes[i][0]:boxes[i][2]]
        text = text_ocr(crop_img)
        result[i] = {'text': text, 'box': boxes[i], 'scores': scores[i]}
    return result

if __name__ == '__main__':
    url = 'https://i.loli.net/2020/03/31/ZimX6ufSL1KxQw9.jpg'
    result = ocr(url)
    print(result)