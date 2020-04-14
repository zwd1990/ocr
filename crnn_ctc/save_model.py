import os
import train
import tensorflow as tf
from inference_crnn_ctc import inference

tf.app.flags.DEFINE_string('export_model_dir', "./output/versions", 'Directory where the model exported files should be placed.')
tf.app.flags.DEFINE_integer('model_version', 72000, 'Models version number.')
FLAGS = tf.app.flags.FLAGS

#传入数据占位符
inputs = tf.placeholder(tf.float32, [None, 256, 32, 3])
seq_len = tf.placeholder(tf.int32, [None])
#前向传播
logits = inference(inputs, seq_len, False)
#解码结果
decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, 
                                                    seq_len, 
                                                    merge_repeated=True)
#模型预测结果
dense_decoded = tf.sparse_tensor_to_dense(
        decoded[0], 
        default_value=-1, 
        name="dense_decoded")
#获取滑动平均参数加载模型
variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)
with tf.Session() as sess:
    #初始化变量
    sess.run((tf.global_variables_initializer(),
                tf.local_variables_initializer()))
    #加载已保存变量
    saver.restore(sess, tf.train.latest_checkpoint('./output/'))
    # Create SavedModelBuilder class
    # defines where the model will be exported
    export_path_base = FLAGS.export_model_dir
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to', export_path)
    export_path = '../tfserving/crnn_ctc/72000'
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # Creates the TensorInfo protobuf objects that encapsulates the input/output tensors
    tensor_info_inputs = tf.saved_model.utils.build_tensor_info(inputs)
    tensor_info_seq_len = tf.saved_model.utils.build_tensor_info(seq_len)
    # output tensor info
    dense_decoded_output = tf.saved_model.utils.build_tensor_info(dense_decoded)

    # Defines the DeepLab signatures, uses the TF Predict API
    # It receives an image and its dimensions and output the segmentation mask
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'inputs': tensor_info_inputs, 'seq_len': tensor_info_seq_len},
            outputs={'result': dense_decoded_output},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature,
        })
    # export the model
    builder.save(as_text=True)
    print('Done exporting!')