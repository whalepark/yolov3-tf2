import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2 ###
import numpy as np
# import tensorflow as tf ###
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs


import logging
import grpc
import sys, os

sys.path.insert(0, os.path.abspath('tfrpc/client'))
import yolo_pb2
import yolo_pb2_grpc
from tf_wrapper import TFWrapper, ControlProcedure
import signal

import pickle ###

CHUNK_SIZE = 4000000 # approximation to 4194304, grpc message size limit

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

g_stub: yolo_pb2_grpc.YoloTensorflowWrapperStub

def initialize(stub):
    global g_stub

    ControlProcedure.Connect(stub)
    g_stub = stub
    signal.signal(signal.SIGINT, finalize)

def finalize():
    ControlProcedure.Disconnect(g_stub)
    
def main(_argv):
    server_addr = os.environ.get('SERVER_ADDR')
    channel = grpc.insecure_channel(f'{server_addr}:1990', \
        options=[('grpc.max_send_message_length', 50 * 1024 * 1024), \
        ('grpc.max_receive_message_length', 50 * 1024 * 1024)] \
    )
    stub = yolo_pb2_grpc.YoloTensorflowWrapperStub(channel)
    initialize(stub)

    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    physical_devices = TFWrapper.tf_config_experimental_list__physical__devices(stub, device_type='GPU')
    if len(physical_devices) > 0: # in my settings, this if statement always returns false
        # tf.config.experimental.set_memory_growth(physical_devices[0], True) 
        TFWrapper.tf_config_experimental_set__memory__growth(physical_devices[0], True) 

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
        # yolo = YoloV3Tiny(stub=stub, classes=FLAGS.num_classes)
    else:
        # yolo = YoloV3(classes=FLAGS.num_classes)
        yolo = YoloV3(stub=stub, classes=FLAGS.num_classes)

    # yolo.load_weights(FLAGS.weights).expect_partial()
    status_obj_id = TFWrapper.attribute_model_load__weight(stub, yolo, FLAGS.weights)
    TFWrapper.attribute_checkpoint_expect__partial(stub, status_obj_id)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        # img_raw = tf.image.decode_image(
        #     open(FLAGS.image, 'rb').read(), channels=3)
        img_raw = TFWrapper.tf_image_decode__image(stub, open(FLAGS.image, 'rb').read(), channels=3)

    # img = tf.expand_dims(img_raw, 0)
    img = TFWrapper.tf_expand__dims(stub, img_raw, 0)
    img = transform_images(stub, img, FLAGS.size)

    t1 = time.time()
    # boxes, scores, classes, nums = yolo(img)
    img_obj_wrapper = yolo_pb2.CallRequest.ObjId()
    img_obj_wrapper.obj_id, img_obj_wrapper.release = img, False
    ret_val = TFWrapper.callable_emulator(stub, yolo, False, 1, img_obj_wrapper)
    ret_val = TFWrapper.iterable_indexing(stub, ret_val[0], 0, iterable_pickled=True)
    boxes, scores, classes, nums = ret_val
    t2 = time.time()
    logging.info('inference_time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    # img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img_raw_numpy = TFWrapper.byte_tensor_to_numpy(stub, img_raw)

    # img = cv2.cvtColor(img_raw_numpy, cv2.COLOR_RGB2BGR)
    # img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    # cv2.imwrite(FLAGS.output, img)

    img_result = cv2.cvtColor(img_raw_numpy, cv2.COLOR_RGB2BGR)
    img_result = draw_outputs(img_result, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(FLAGS.output, img_result)

    logging.info('output saved to: {}'.format(FLAGS.output))

    finalize()


if __name__ == '__main__':
    try:
        logging.basicConfig()
        app.run(main)
    except SystemExit:
        pass
