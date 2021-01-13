#!/usr/bin/python
import socket
import logging
from absl import flags
from absl.flags import FLAGS
import numpy as np
import sys
from pocketmgr import PocketManager



yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

flags.DEFINE_integer('yolo_max_boxes', 100,
                     'maximum number of boxes per image')
flags.DEFINE_float('yolo_iou_threshold', 0.5, 'iou threshold')
flags.DEFINE_float('yolo_score_threshold', 0.5, 'score threshold')

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

##### Misun Defined
PERF_SERVER_SOCKET = '/sockets/perf_server.sock'

def make_json(container_id):
    import json
    args_dict = {}

    args_dict['type']='closed-proc-ns'
    args_dict['cid']=container_id
    args_dict['events']=['cpu-cycles','page-faults','minor-faults','major-faults','cache-misses','LLC-load-misses','LLC-store-misses','dTLB-load-misses','iTLB-load-misses','instructions']

    args_json = json.dumps(args_dict)

    return args_json

def connect_to_perf_server(container_id: str):
    logging.info('connect_to_perf_server')
    my_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    my_socket.connect(PERF_SERVER_SOCKET)
    json_data_to_send = make_json(container_id)
    my_socket.sendall(json_data_to_send.encode('utf-8'))
    data_received = my_socket.recv(1024)
    logging.info(f'data_received={data_received}')
    my_socket.close()

if __name__ == '__main__':
    FLAGS(sys.argv)
    mgr = PocketManager()
    connect_to_perf_server(socket.gethostname())
    mgr.start()
