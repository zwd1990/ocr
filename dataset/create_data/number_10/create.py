# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:09:29 2020

@author: JianjinL
"""
import os
import cv2
import random
import json
import numpy as np
from PIL import Image, ImageFont, ImageDraw

#待训练字符集
DIGITS = {0:'0', 1:'1', 2:'2',3: '3',4: '4',5: '5',6: '6',7: '7',8: '8',9: '9'}

def gen_text(data_dir, train_num):
    '''
    随机生成不定长样本数据
    :param data_dir: 生成样本保存的文件夹5
    :param train_num: 生成样本数据的数量
    :return:
    '''

    labels = {}
    #循环生成样本图片
    for i in range(train_num):
        #加载字体文件位置
        font_path = '../font/' + random.choice(os.listdir('../font/'))
        #字体大小
        font_size = random.randint(20, 40)
        #创建一个字体对象
        font = ImageFont.truetype(font_path, font_size)
        #创建一个字典保存标签
        #随机生成文本的长度,最大长度为10
        rnd = random.randint(1, 10)
        #创建承装文本的容器，初始化为空字符串
        text = ''
        label = []
        #随机生成指定文本长度的字符集
        for j in range(rnd):
            #随机取得字符集中的一个元素，加入容器中
            randnum = random.randint(0, len(DIGITS)-1)
            text = text + DIGITS[randnum]
            label.append(randnum)
        #新建一个图像对象，mode="RGB",size=(256,32),color=随机值
        red_image = random.randint(0, 255)
        yellow_image = random.randint(0, 255)
        blue_image = random.randint(0, 255)
        text_x,text_y = (random.randint(0,50), random.randint(0,20))
        #图片大小
        images = (text_x+ rnd*22 + random.randint(0, 2), text_y+font_size+random.randint(5, 10))
        img = Image.new("RGB", images, 'white')
        #创建画图接口
        draw = ImageDraw.Draw(img)
        #将生成的文本写入图片
        red_text = random.randint(0, 255)
        yellow_text = random.randint(0, 255)
        blue_text = random.randint(0, 255)
        draw.text((text_x,text_y), text, font=font, fill=(red_text, yellow_text, blue_text))
        #将图片转换为数组对象
        img=np.array(img)
        #向图片中加入椒盐噪声，模拟真实环境
        #img = img_salt_pepper_noise(img, float(random.randint(1,5)/100.0))
        #将图片写入文件夹中
        cv2.imwrite(data_dir + str(i+1) + '.jpg',img)
        labels[i+1] = label
    #将图片序号与标签对映关系写入json中
    with open(data_dir+"labels.json","w",encoding='utf-8') as f:
        json.dump(labels,f)
            
def img_salt_pepper_noise(src, percetage):
    '''
    为图片加入椒盐噪声
    :param src: 图片数组对象
    :param percetage: 加入噪声的比例
    :return NoiseImg: 加入椒盐噪声后的图片数组对象
    '''
    NoiseImg = src
    #噪声数量
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    #将噪声点随机加入图片中
    for i in range(NoiseNum):
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)
        if random.randint(0,1) == 0:
            NoiseImg[randX,randY] = random.randint(0, 255)
        else:
            NoiseImg[randX,randY] = random.randint(0, 255)
    return NoiseImg
    
if __name__ == '__main__':
    #生成训练集
    gen_text('../../images/number_10/train/', 32000)
    #生成验证集
    gen_text('../../images/number_10/test/', 3200)
    #生成测试集
    gen_text('../../images/number_10/validation/', 3200)