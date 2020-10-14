
def make_json(container_id):
    import json
    args_dict = {}

    args_dict['type']='closed-proc-ns'
    args_dict['cid']=container_id
    args_dict['events']=['cpu-cycles','page-faults','minor-faults','major-faults','cache-misses','LLC-load-misses','LLC-store-misses','dTLB-load-misses','iTLB-load-misses']

    args_json = json.dumps(args_dict)

    return args_json

def connect_to_perf_server():
    import socket
    PERF_SERVER_SOCKET = '/sockets/perf_server.sock'
    container_id = socket.gethostname()
    my_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    my_socket.connect(PERF_SERVER_SOCKET)
    json_data_to_send = make_json(container_id)
    my_socket.sendall(json_data_to_send.encode('utf-8'))
    data_received = my_socket.recv(1024)
    print(data_received)
    my_socket.close()

connect_to_perf_server()

import time, subprocess
import shlex # for subprocess popen
from multiprocessing import Process
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

import sysv_ipc

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

# Misun defined
PERF_SERVER_SOCKET = '/sockets/perf_server.sock'
flags.DEFINE_boolean('hello', False, 'hello or health check')
flags.DEFINE_string('object', 'path', 'specify how to pass over objects')
flags.DEFINE_integer('num_images', 1, 'the number of images to process')
flags.DEFINE_integer('size_to_transfer', 4*1024*1024, 'the size of image')

g_stub: yolo_pb2_grpc.YoloTensorflowWrapperStub
# g_redis: redis.Redis
CONTAINER_ID: str
shmem = None

class SharedMemoryChannel:
    def __init__(self, key, size):
        self.shmem = sysv_ipc.SharedMemory(key, sysv_ipc.IPC_CREX, size=size)
        self.key = key
        self.sem = sysv_ipc.Semaphore(key, sysv_ipc.IPC_CREX)

    def write(self, uri):
        self.sem.acquire()
        self.shmem.write(open(uri, 'rb').read())
        self.sem.release()

    def read(self, size):
        self.sem.acquire()
        data = self.shmem.read(size)
        self.sem.release()
        return data

    def view(self, size):
        self.sem.acquire()
        mv = memoryview(self.shmem)
        self.sem.release()
        return mv

    def finalize(self):
        self.shmem.detach()
        self.shmem.remove()
        self.sem.remove()



def initialize(stub, server_addr, data_channel):
    global g_stub, CONTAINER_ID

    container_id = subprocess.check_output('cat /proc/self/cgroup | grep cpuset | cut -d/ -f3 | head -1', shell=True, encoding='utf-8').strip()
    config_data_channel(data_channel, int(container_id[:8], 16))

    ControlProcedure.Connect(stub, FLAGS.object, container_id, shmem_channel=shmem) # path, bin, redis, shmem
    g_stub = stub
    signal.signal(signal.SIGINT, finalize)
    CONTAINER_ID=container_id

def config_data_channel(data_channel, key = None):
    if data_channel == 'path':
        pass
    elif data_channel == 'bin':
        pass
    elif data_channel == 'redis':
        # put_in_redis(FLAGS.image)
        pass
    elif data_channel == 'shmem':
        global shmem
        shmem = SharedMemoryChannel(key=key, size=FLAGS.num_images * FLAGS.size_to_transfer)

def finalize():
    ControlProcedure.Disconnect(g_stub)
    shmem.finalize()

def put_in_redis(image_path):
    with open(image_path, 'rb') as f:
        image_bin = f.read()
    
def main(_argv):
    # os.environ['SERVER_ADDR'] = 'localhost' # todo: remove after debugging
    server_addr = os.environ.get('SERVER_ADDR')
    channel = grpc.insecure_channel(f'{server_addr}:1990', \
        options=[('grpc.max_send_message_length', 10 * 1024 * 10), \
        ('grpc.max_receive_message_length', 10 * 1024 * 100), \
        ('grpc.max_message_length', 10 * 1024 * 100)] \
    )
    stub = yolo_pb2_grpc.YoloTensorflowWrapperStub(channel)
    initialize(stub, server_addr, FLAGS.object)

    if FLAGS.hello:
        health = ControlProcedure.SayHello(stub, 'misun')
        exit()

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
    status_obj_id = TFWrapper.attribute_model_load__weights(stub, 'yolov3', FLAGS.weights) ## todo check if already weighted
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
        start=time.time()
        if FLAGS.object == 'bin':
            img_raw = TFWrapper.tf_image_decode__image(stub, 
                channels=3, data_channel=FLAGS.object, data_bytes=open(FLAGS.image, 'rb').read())
        elif FLAGS.object == 'path':
            img_raw = TFWrapper.tf_image_decode__image(stub, image_path=FLAGS.image, 
                channels=3, data_channel=FLAGS.object)
        elif FLAGS.object == 'shmem':
            img_raw = TFWrapper.tf_image_decode__image(stub, image_path=FLAGS.image, 
                channels=3, data_channel=FLAGS.object, data_size_in_byte=FLAGS.size_to_transfer)
        else:
            raise Exception(f'Unknown data channel={FLGAS.object}')
        end=time.time()
        logging.info(f'time={end-start}')

    # img = tf.expand_dims(img_raw, 0)
    img = TFWrapper.tf_expand__dims(stub, img_raw, 0)
    img = transform_images(stub, img, FLAGS.size)

    t1 = time.time()
    # boxes, scores, classes, nums = yolo(img)
    img_obj_wrapper = yolo_pb2.CallRequest.ObjId()
    img_obj_wrapper.obj_id, img_obj_wrapper.release = img, False
    ret_val = TFWrapper.callable_emulator(stub, yolo, True, 1, 'yolov3', img_obj_wrapper)
    ret_val = TFWrapper.iterable_indexing(stub, ret_val, 0)
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
