# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:59:51 2020

@author: JianjinL
"""


import tensorflow as tf
from inference_crnn_ctc import inference

#每批数据的数量
BATCH_SIZE = 64
#文件位置
tfrecord_dir = "../dataset/tfrecord/english_train.tfrecords"
save_dir = "./output/english/crnn_ctc_model.ctpk"
save_path = '/'.join(save_dir.split('/')[:-1])+'/'
#模型参数
INITIAL_LEARNING_RATE = 1e-3 # 初始化学习速率
DECAY_STEPS = 5000
REPORT_STEPS = 20
LEARNING_RATE_DECAY_FACTOR = 0.9
MOMENTUM = 0.9
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
shuffle_buffer = 1600  #定义随机打乱数据时buffer的大小。
NUM_EPOCHS = 50#所有样本数据训练轮数
#重置计算图
tf.reset_default_graph()

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

def train():
    '''
    模型训练过程，总共分为三个步骤。
    1、数据读取，获取一个batch的数据传入模型
    2、定义模型结构损失函数，优化方法
    3、开始训练并保存模型
    '''
    #获取训练数据输入文件
    train_files = tf.train.match_filenames_once(tfrecord_dir)
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
    logits = inference(
            inputs, 
            seq_len, 
            True)
    #迭代轮数计数器
    global_step = tf.Variable(0, trainable=False)
    #滑动平均
    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, 
            global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    #定义学习率
    learning_rate = tf.train.exponential_decay(
            INITIAL_LEARNING_RATE,
            global_step,
            DECAY_STEPS,
            LEARNING_RATE_DECAY_FACTOR,
            staircase=True
            )
    #设置ctc损失函数
    loss_ctc = tf.reduce_mean(tf.nn.ctc_loss(
            labels=targets, 
            inputs=logits, 
            sequence_length=seq_len,
            preprocess_collapse_repeated=True, 
            ctc_merge_repeated=False,
            ignore_longer_outputs_than_inputs=True))
    #设置优化器
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_ctc, global_step=global_step)
    #解码结果
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, 
                                                      seq_len, 
                                                      merge_repeated=False)
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
    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss_ctc", loss_ctc)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", 1-acc)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    #配置GPU
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    #创建会话
    with tf.Session(config=sess_config) as sess:
        # 将网络结构图写到文件中
        writer = tf.summary.FileWriter(
            './logs/'+save_dir.split('/')[2] + '/', 
            graph=tf.get_default_graph())
        try:
            #初始化变量
            sess.run((tf.global_variables_initializer(),
                      tf.local_variables_initializer()))
            #加载已保存变量
            saver.restore(sess, tf.train.latest_checkpoint(save_path))
            print('模型加载成功，即将开始继续训练。')
            continuetrain = True
        except:
            #初始化变量
            sess.run((tf.global_variables_initializer(),
                      tf.local_variables_initializer()))
            print('模型加载失败，即将开始重新训练。')
            continuetrain = False
        # 初始化训练数据的迭代器。
        sess.run(iterator.initializer)
        #训练轮数计数器
        if continuetrain:
            ckpt = tf.train.get_checkpoint_state(save_path)
            training_steps = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            training_steps = 0
        while True:
            try:
                #进行迭代计算，记录当前拟合结果
                if training_steps % 1000 != 0:
                    (global_step_, 
                    train_step_, 
                    loss_ctc_) = sess.run(
                            [global_step, 
                            train_step, 
                            loss_ctc])
                #每1000轮输出当前的表现，并保存模型
                else:
                    (global_step_, 
                    train_step_, 
                    loss_ctc_,
                    acc_,
                    dense_decoded_,
                    labels_,
                    summary) = sess.run(
                            [global_step, 
                            train_step, 
                            loss_ctc,
                            acc,
                            dense_decoded,
                            labels,
                            merged_summary_op])
                    print('''training_steps = {0}\ncost = {1}\naccuracy = {2}\nmodel_labels = \n{3}\ntrue_labels = \n{4}
                            '''.format(
                            training_steps, 
                            loss_ctc_,
                            (1-acc_),
                            dense_decoded_,
                            labels_))
                    saver.save(
                            sess, 
                            save_dir, 
                            global_step=training_steps)
                    # Write logs at every iteration
                    writer.add_summary(summary, training_steps)
                #训练完一轮计数器加一
                training_steps += 1
            except tf.errors.OutOfRangeError:
                break

if __name__ == '__main__':
    train()
            