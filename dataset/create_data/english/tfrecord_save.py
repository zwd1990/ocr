# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:18:33 2020

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import json
import os
import cv2

# 图片大小
OUTPUT_SHAPE = (256, 32, 3)

# 定义函数转化变量类型。
#生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
#生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 将数据转化为tf.train.Example格式。
def _make_example(images, labels, seq_len):
    example = tf.train.Example(features=tf.train.Features(feature={
        'images': _bytes_feature(images),#图片
        'labels':  _int64_feature(labels),#标签，稀疏矩阵
        'seq_len': _int64_feature(seq_len)#标签长度
    }))
    return example

def get_file_text_array(data_dir):
    '''
    获取所有图片文件名及对应图片标签
    :param data_dir: 获取数据的文件夹地址
    '''
    file_name_array=[]
    #获取图片文件列表
    with open(data_dir) as f:
        data = f.readlines()
    file_name_array = [img.split(' ')[0][1:] for img in data]
    return file_name_array

def get_file_label(data_dir):
    '''
    获取所有图片文件名及对应图片标签
    :param data_dir: 获取数据的文件夹地址
    '''
    labelList=[]
    #获取图片文件列表
    with open(data_dir+'') as f:
        data = f.readlines()
    labelList = [label.replace('\n','') for label in data]
    return labelList

def create_tfrecord(data_dir, save_path, data_type):
    '''

    '''
    # 读取图片训练数据。
    file_name_array = get_file_text_array(data_dir+'annotation_'+data_type+'.txt')
    #标签列表
    labelList = get_file_label(data_dir+'lexicon.txt')
    with open("labels.json","r") as f:
        dic = json.load(f)
    # 输出包含训练数据的TFRecord文件。
    with tf.python_io.TFRecordWriter(save_path) as writer:
        #遍历所有图片
        for i in range(len(file_name_array)):
            try:
                #读取文件对象
                image = cv2.imread(data_dir + file_name_array[i])
                #改变图片大小
                image = cv2.resize(image, (OUTPUT_SHAPE[0], OUTPUT_SHAPE[1]), 3)
                #将图片从BGR格式转换成灰度图片
                #image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
                #将图片转为narray格式
                image = image.reshape(OUTPUT_SHAPE)
                image = image.tostring()
                #标签列表
                label = [dic[char] for char in labelList[int(file_name_array[i].split(' ')[0][1:].split('_')[2].replace('.jpg',''))]]
                #序列长度
                seq_len = np.asarray([OUTPUT_SHAPE[0]])
                #将数据转化为tf.train.Example格式
                example = _make_example(image, label, seq_len)
                #写入文件
                writer.write(example.SerializeToString())
            except:
                print('该图片写入出错。')
    print(save_path+"文件已保存。")

if __name__ == '__main__':
    create_tfrecord('../../images/english/mnt/ramdisk/max/90kDICT32px/', "../../tfrecord/english_train.tfrecords",'train')
    create_tfrecord('../../images/english/mnt/ramdisk/max/90kDICT32px/', "../../tfrecord/english_test.tfrecords",'test')
    create_tfrecord('../../images/english/mnt/ramdisk/max/90kDICT32px/', "../../tfrecord/english_validation.tfrecords",'val')