# coding=utf-8
import os
import shutil
import sys
import time
import network
import cv2
import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector

tf.app.flags.DEFINE_string('test_data_path', 'data/demo/', '')
tf.app.flags.DEFINE_string('output_path', 'data/res/', '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_mlt/', '')
#新加入的
tf.app.flags.DEFINE_string('export_model_dir', "../tfserving/ctpn", 'Directory where the model exported files should be placed.')
tf.app.flags.DEFINE_integer('model_version', 50000, 'Models version number.')
FLAGS = tf.app.flags.FLAGS

def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


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


def main(argv=None):
    if os.path.exists(FLAGS.output_path):
        shutil.rmtree(FLAGS.output_path)
    os.makedirs(FLAGS.output_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)
#############################################################下面为新的代码
            # Create SavedModelBuilder class
            # defines where the model will be exported
            export_path_base = FLAGS.export_model_dir
            export_path = os.path.join(
                tf.compat.as_bytes(export_path_base),
                tf.compat.as_bytes(str(FLAGS.model_version)))
            print('Exporting trained model to', export_path)
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            # Creates the TensorInfo protobuf objects that encapsulates the input/output tensors
            tensor_info_input = tf.saved_model.utils.build_tensor_info(input_image)

            # output tensor info
            bbox_pred_output = tf.saved_model.utils.build_tensor_info(bbox_pred)
            cls_pred_output = tf.saved_model.utils.build_tensor_info(cls_pred)
            cls_prob_output = tf.saved_model.utils.build_tensor_info(cls_prob)

            # Defines the DeepLab signatures, uses the TF Predict API
            # It receives an image and its dimensions and output the segmentation mask
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': tensor_info_input},
                    outputs={'bbox_pred_output': bbox_pred_output,'cls_pred_output':cls_pred_output,'cls_prob_output':cls_prob_output},
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

if __name__ == '__main__':
    tf.app.run()
