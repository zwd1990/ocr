# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:59:51 2020

@author: JianjinL
"""

import tensorflow as tf
from inference_crnn_ctc import inference
import train
import time

BATCH_SIZE = 32 #每一批的数据量
shuffle_buffer = 16  #定义随机打乱数据时buffer的大小。
NUM_EPOCHS = 100#所有样本数据训练轮数

def parser(record):
    '''
    解析TFRecord文件的方法
    :return images: 图片矩阵 256 * 32 * 3
    :return labels: 图片标签 稀疏矩阵
    :return seq_len: 序列长度 256
    '''
    features = tf.parse_single_example(
        record,
        features={
            'images':tf.FixedLenFeature([],tf.string),
            'labels':tf.VarLenFeature(tf.int64),
            'seq_len':tf.FixedLenFeature([],tf.int64)
        })
    
    images = tf.decode_raw(features['images'], tf.uint8)
    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, [256,32,3])
    labels = tf.cast(features['labels'], tf.int32)
    seq_len = tf.cast(features['seq_len']/17, tf.int32)
    return images, labels, seq_len

def predict():
    '''
    模型训练过程
    '''
    #获取训练数据输入文件
    train_files = tf.train.match_filenames_once(
            "../dataset/tfrecord/english_test.tfrecords")
    #定义读取训练数据的数据集
    dataset = tf.data.TFRecordDataset(train_files)
    #解析数据集
    dataset = dataset.map(parser)
    # 对数据进行shuffle和batching操作。
    dataset = dataset.shuffle(shuffle_buffer).batch(BATCH_SIZE)
    # 重复NUM_EPOCHS个epoch。
    dataset = dataset.repeat(NUM_EPOCHS)
    # 定义数据集迭代器。
    iterator = dataset.make_initializable_iterator()
    #获取一个batch的训练数据
    inputs, targets, seq_len = iterator.get_next()
    #前向传播
    logits = inference(inputs, seq_len)
    #设置ctc损失函数
    loss_ctc = tf.reduce_mean(tf.nn.ctc_loss(
            labels=targets, 
            inputs=logits, 
            sequence_length=seq_len))
    #解码结果
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, 
                                                      seq_len, 
                                                      merge_repeated=True)
    #模型预测结果
    dense_decoded = tf.sparse_tensor_to_dense(
            decoded[0], 
            default_value=-1, 
            name="dense_decoded")
    #输入标签
    labels = tf.sparse_tensor_to_dense(targets,
                                       default_value=-1, 
                                       name="labels")
    #错误率
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
    saver = tf.train.Saver()
    #创建会话
    while True:
        with tf.Session() as sess:
            #初始化变量
            sess.run((tf.global_variables_initializer(),
                      tf.local_variables_initializer()))
            #加载已保存变量
            saver.restore(sess, tf.train.latest_checkpoint('./output/english/'))
            print('模型加载成功，准备开始测试...')
            # 初始化训练数据的迭代器。
            sess.run(iterator.initializer)
            #当前训练轮数
            ckpt = tf.train.get_checkpoint_state('./output/english/')
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            loss_score, accuracy, dense_decoded_,labels_ = sess.run([loss_ctc, acc,dense_decoded,labels])
            print('step:{0}\nloss_score：{1}\naccuracy:{2}\nresult:\n{3}\ntargets:\n{4}'.format(global_step,loss_score,1-accuracy,dense_decoded_,labels_))
        time.sleep(60*3)
        
if __name__ == '__main__':
    predict()
