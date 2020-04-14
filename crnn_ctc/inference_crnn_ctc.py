# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:42:22 2020

@author: JianjinL
"""

import tensorflow as tf

#输出层神经元个数
NUM_CLASSES = 37
#加入slim
slim = tf.contrib.slim

def inference(input_tensor, seq_len , is_train = True):
    '''
    模型的前向传播过程，CNN+BLSTM+CTC
    :parma input_tensor: batch_size * 256 * 32 * 1
    '''
    #CNN层
    with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            kernel_size=3,
                            stride=1,
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            biases_initializer=None):
        net = slim.conv2d(input_tensor, 64, scope='layer1-conv')
        net = slim.max_pool2d(net, 2, 2, scope='layer2-max-pool')
        net = slim.conv2d(net, 128, scope='layer3-conv')
        net = slim.max_pool2d(net, 2, 2, scope='layer4-max-pool')
        net = slim.conv2d(net, 256, scope='layer5-conv')
        net = slim.conv2d(net, 256, scope='layer6-conv')
        net = slim.max_pool2d(net, [1,2], 2, scope='layer7-max-pool')
        net = slim.conv2d(net, 512, scope='layer8-conv')
        net = slim.batch_norm(net, decay=0.999, scope='layer9-bn')
        net = slim.conv2d(net, 512, scope='layer10-conv')
        net = slim.batch_norm(net, decay=0.999, scope='layer11-bn')
        net = slim.max_pool2d(net, [1,2], 2, scope='layer12-max-pool')
        net = slim.conv2d(net, 512,kernel_size=2,padding='VALID', scope='layer13-conv')
    #改变CNN输出的张量维度
    with tf.variable_scope('map_to_sequence'):
        shape = net.get_shape().as_list()
        assert shape[2] == 1
        ret = tf.squeeze(net, axis=2, name='squeeze')
    #接上一个BLSTM层
    with tf.variable_scope('BLSTM'):
        # 使用多层的LSTM结构。
        lstm_cell_fw = [tf.nn.rnn_cell.LSTMCell(nh, forget_bias=1.0) for nh in [256] * 2]
        lstm_cell_bw = [tf.nn.rnn_cell.LSTMCell(nh, forget_bias=1.0) for nh in [256] * 2]
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            lstm_cell_fw, 
            lstm_cell_bw, 
            ret,
            dtype=tf.float32)
        #bilstm_outputs = tf.concat([outputs[0], outputs[1]],-1)
        bilstm_outputs = tf.reshape(outputs, [-1, 512])
    #接一个全连接层
    with tf.variable_scope('fc'):
        weights = tf.get_variable("weight", [512, NUM_CLASSES],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("bias", [NUM_CLASSES], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(bilstm_outputs, weights) + biases
    if not is_train:
        BATCH_SIZE = 1
    else:
        BATCH_SIZE = tf.shape(input_tensor)[0]
    logits = tf.reshape(logit, [BATCH_SIZE, -1, NUM_CLASSES])
    # 转置矩阵， 256 * BATCH_SIZE * 11
    logits = tf.transpose(logits, (1, 0, 2))
    return logits

