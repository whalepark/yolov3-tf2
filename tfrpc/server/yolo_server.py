#!/usr/bin/python
import os
import subprocess, time # todo: subprocess: remove?
from multiprocessing import Process # todo: remove?

# todo: remove these after debugging
# os.environ['YOLO_SERVER'] = '1'
from multiprocessing import Process, Pipe, Manager
# import time
# import contexttimer
def child_process(func):
    """Makes the function run as a separate process."""
    def wrapper(*args, **kwargs):
        def worker(conn, func, args, kwargs):
            conn.send(func(*args, **kwargs))
            conn.close()
        parent_conn, child_conn = Pipe()
        p = Process(target=worker, args=(child_conn, func, args, kwargs))
        p.start()
        ret = parent_conn.recv()
        p.join()
        return ret
    return wrapper

from concurrent import futures
import logging
import grpc

import yolo_pb2, yolo_pb2_grpc
import pickle

from absl import flags
from absl.flags import FLAGS

import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
)

from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)

import sys
cwd = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, os.path.abspath('../../yolov3_tf2'))
os.chdir('../..')
sys.path.insert(0, os.path.abspath('yolov3_tf2'))
print(os.getcwd()) ##
from batch_norm import BatchNormalization
from utils import broadcast_iou
import threading
from enum import Enum
# os.chdir(cwd)
# from collections.abc import Iterable
# import inspect


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


## global variables
Model_Create_Lock = threading.Lock()
Weights_Load_Lock = threading.Lock()
Graph_Build_In_Progress = False
Connection_Id = "";

Global_Tensor_Dict = {}
Object_Ownership = {}
Connection_Set = set()
# Global_Graph_Dict = {}
# Global_Sess_Dict = {}
# Global_Model_Dict = Manager().dict()
Global_Model_Dict = {}
Container_Id_Dict = {}
Subdir_Dict = {}
hostroot = '/hostroot'
subdir_root = hostroot + '/var/lib/docker/overlay2'

conv2d_count = 0
batch_norm_count = 0
leaky_re_lu_count = 0
zero_padding2d_count=0
add_count = 0
lambda_count = 0

class ModelInfo:
    class State(Enum):
        BUILD_IN_PROGRESS = 1
        BUILD_DONE = 2
        LOAD_WEIGHT_DONE = 3

    def __init__(self, name, connection_id):
        self.name = name
        self.connection_id = connection_id
        self.state = ModelInfo.State.BUILD_IN_PROGRESS
        self.obj_id = -1
        self.is_weight_loaded = False
        self.weight_checkpoint = None

    def set_done(self, model):
        global Graph_Build_In_Progress, Connection_Id

        self.state = ModelInfo.State.BUILD_DONE
        Graph_Build_In_Progress = False
        Connection_Id = ""
        self.model = model
        self.obj_id = id(model)

    def is_done(self):
        return self.state

    def load_weights(self, model, weights_path):
        if self.is_weight_loaded:
            return self.weight_checkpoint
        else:
            with Weights_Load_Lock:
                self.weight_checkpoint = model.load_weights(weights_path)
                self.state = ModelInfo.State.LOAD_WEIGHT_DONE
                self.is_weight_loaded = True
            return self.weight_checkpoint

    def is_weight_loaded(self, model):
        return self.is_weight_loaded

class IllegalModelBuildException(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return 'IllegalModelBuildException'

class PocketModel(tf.keras.Model):
    def call(self, inputs, training=False, sendpipe=None):
        # # print(type(self))
        # # print(inputs)
        # ret_val = super(tf.keras.Model, self).__call__(inputs, training)
        # # print('there')
        # return ret_val

        ret_val = super().call(inputs, training)
        if sendpipe != None:
            sendpipe.send(ret_val)
            sendpipe.close()
        return ret_val

class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def utils_get_obj(obj_id: int):
    return Global_Tensor_Dict[obj_id]

def utils_set_obj(obj, connection_id):
    obj_id: int = id(obj)
    Global_Tensor_Dict[obj_id] = obj
    if connection_id in Object_Ownership:
        Object_Ownership[connection_id].append(obj_id)
    else:
        Object_Ownership[connection_id] = [obj_id]
    return obj_id

def utils_rm_obj(obj_ids):
    for obj_id in obj_ids:
        del Global_Tensor_Dict[obj_id]

def utils_is_iterable(obj):
    try:
        _ = (e for e in obj)
        return True
    except TypeError:
        #  my_object, 'is not iterable'
        return False

def utils_convert_elem_into_array(source: list, destination: list):
    destination[0] = [None for _ in range(len(source))]
    new_iterable = destination[0]
    for index in range(len(source)):
        if isinstance(source[index], (list, tuple)):
            new_iterable[index]=[]
            utils_convert_elem_into_array(source[index], new_iterable[index])
        elif isinstance(source[index], tf.Tensor):
            new_iterable[index] = source[index].numpy()
        else:
            new_iterable[index] = source[index]

def utils_collect_garbage(connection_id: str):
    global Global_Tensor_Dict

    items_to_remove = Object_Ownership[connection_id]
    for item in items_to_remove:
        del Global_Tensor_Dict[item]
    del Subdir_Dict[connection_id]

def utils_inference_wrapper(sndpipe, model, args):
    print(f'misun: 0')
    print('misun:', model, type(args), args)
    ret_val = model(args)
    print(f'misun: 1')
    sndpipe.send(ret_val)
    print(f'misun: 2')
    sndpipe.close()
    print(f'misun: 3')

def utils_infer_target(snd, callable_obj, args):
    obj_name = Global_Model_Dict[callable_obj].model
    ret_val = obj_name(args)
    # ret_val = callable_obj(args)
    snd.send(ret_val)
    return 0

def _get_client_root(container):
    # lower_list = _get_all_subdir(container)
    # subdir_merged = _merge_subdir(lower_list)
    # subdir_root += _subdir_merged
    # try:
    #     output = subprocess.check_output('docker info', shell=True).decode('utf-8').strip()
    # except subprocess.CalledProcessError as e:
    #     print(e.output)
    merged_dir = subprocess.check_output('docker inspect -f {{.GraphDriver.Data.MergedDir}} ' + container, shell=True).decode('utf-8').strip()
    print(f'merged_dir={merged_dir}')
    return hostroot + merged_dir

def utils_add_to_subdir(container_id, connection_id):
    subdir = _get_client_root(container_id)
    print(f'subdir={subdir}')
    output = subprocess.check_output(f'ls {subdir}', shell=True).decode('utf-8').strip()
    print(f'output={output}')
    Subdir_Dict[connection_id] = subdir


# @jit(nopython=True, nogil=True, parallel=True)
# @ray.remote
@child_process
def utils_infer(callable_obj, args):
    return callable_obj(args)

def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box

def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=FLAGS.yolo_max_boxes,
        max_total_size=FLAGS.yolo_max_boxes,
        iou_threshold=FLAGS.yolo_iou_threshold,
        score_threshold=FLAGS.yolo_score_threshold
    )

    return boxes, scores, classes, valid_detections

class YoloFunctionWrapper(yolo_pb2_grpc.YoloTensorflowWrapperServicer):
    def ModelBuildingAPI(decoratee_func):
        def decorator_func(self, *args, **kwargs):
            request = args[0]
            if request.connection_id != Connection_Id:
                print(Color.RED + 'ERROR: Connection ID mismatch!' + Color.END)
                print(f'request.connection_id={request.connection_id}, Connection_Id={Connection_Id}')
                raise IllegalModelBuildException
            else: 
                return self.decoratee_func(*args, **kwargs)
        return decoratee_func

    def Connect(self, request, context):
        print(f'\nConnect: {request.id}')
        response = yolo_pb2.ConnectResponse()

        if request.id in Connection_Set:
            response.accept = False
        else:
            response.accept = True
            Connection_Set.add(request.id)
            utils_add_to_subdir(request.container_id, request.id)
            # Global_Graph_Dict[request.id] = tf.Graph()
            # Global_Sess_Dict[request.id] = tf.compat.v1.Session(graph=Global_Graph_Dict[request.id])
          
        return response

    def Disconnect(self, request, context):
        print(f'\nDisconnect: {request.id}')
        response = yolo_pb2.DisconnectResponse()

        Connection_Set.remove(request.id)
        # Global_Sess_Dict[request.id].close()
        # del Global_Graph_Dict[request.id]
        # del Global_Sess_Dict[request.id]
        ## todo: global model dict
        threading.Thread(target = utils_collect_garbage, args=[request.id]).start()

        return response

    def CheckIfModelExist(self, request, context):
        global Graph_Build_In_Progress, Connection_Id

        print(f'\CheckIfModelExist')
        response = yolo_pb2.CheckModelExistResponse()

        if request.name in Global_Model_Dict:
            if Global_Model_Dict[request.name].state is ModelInfo.State.BUILD_IN_PROGRESS:
                response.exist = True
                response.model_obj_id = 0
            else:
                response.exist = True
                response.model_obj_id = Global_Model_Dict[request.name].obj_id
        else: # no model, give this connection the lock
            if request.plan_to_make:
                with Model_Create_Lock:
                    Graph_Build_In_Progress = True
                    Connection_Id = request.connection_id
                    Global_Model_Dict[request.name] = ModelInfo(request.name, request.connection_id)
            response.exist = False
            response.model_obj_id = 0
    
        return response

    
    def SayHello(self, request, context):
        return yolo_pb2.HelloReply(message=f'Hello, {request.name}')

    def callable_emulator(self, request, context):
        print(f'\ncallable_emulator')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():
        # with tf.variable_scope(request.connection_id, reuse=True):
        response = yolo_pb2.CallResponse()
        
        callable_obj_id = request.callable_obj_id
        callable_obj = None
        temp_args = []
        del_list = []

        ret_val = []
        ret_val1: int
        ret_val2: int
        ret_val3: int
        ret_val4: int

        for arg in request.obj_ids:
            obj = utils_get_obj(arg.obj_id)
            temp_args.append(obj)
            if arg.release:
                del_list.append(arg.obj_id)

        if len(temp_args) > 1:
            args = temp_args
        else:
            args = temp_args[0]
        
        if request.inference:
            callable_obj = Global_Model_Dict[request.callable_model_name].model
        else:
            callable_obj = utils_get_obj(callable_obj_id)
        print(f'callable={type(callable_obj)}\nname={callable_obj.name}\narg_type={type(args)}')

        if request.num_of_returns == 1:
            # ret_val1 = callable_obj(args)
            if request.inference:
                # utils_infer(callable_obj, args)
                # with contexttimer.Timer(time.perf_counter) as pure_wall:
                #     with contexttimer.Timer(time.process_time) as pure_cpu:
                ret_val1 = callable_obj(args)
                # print(f'misun: wall_time={pure_wall.elapsed}, cpu_time={pure_cpu.elapsed}')
                # ray.get(utils_infer.remote(callable_obj, args))
                # snd, rcv = Pipe()
                # listed_args = [snd, request.callable_model_name, args]
                # worker = Process(target=utils_infer_target, args=listed_args)
                # worker.start()
                # ret_val1 = rcv.recv()
                # worker.join()

                ### 1
                # # worker = Process(target=callable_obj, args=args)
                # # print(args)
                # snd, rcv = Pipe()
                # listed_args = [snd]
                # listed_args.append(callable_obj)
                # listed_args.append(args)
                # # for elem in args:
                # #     print(f'misun: elem={type(elem)}')
                # #     listed_args.append(elem)
                # # listed_args = [snd, callable_obj] + args
                # worker = Process(target=utils_inference_wrapper, args=listed_args)
                # worker.start()
                # worker.join()
                # print('misun: here?')
                # ret_val1 = rcv.recv()


                ### 2: kinda okay
                # worker = Process(target=callable_obj, args=args)
                # worker.start()
                # worker.join()


                ### 2-1
                # worker = Process(target=callable_obj, args=[args])
                # worker.start()
                # worker.join()

                ### 3: ok
                # ret_val1 = callable_obj([args])
                
                ### 4:ok
                # ret_val1 = callable_obj(args)

                ### 5: ok
                # import threading
                # worker = threading.Thread(target=callable_obj, args=args)
                # worker.start()
                # worker.join()

                ### 5-1: ok (thread)
                # import threading, multiprocessing
                # snd, rcv = Pipe()
                # listed_args = [snd]
                # listed_args.append(callable_obj)
                # listed_args.append(args)
                # worker = threading.Thread(target=utils_inference_wrapper, args=listed_args)
                # multiprocessing.set_start_method('spawn')
                # # worker = Process(target=utils_inference_wrapper, args=listed_args)
                # worker.start()
                # ret_val1 = rcv.recv()
                # worker.join()
                # print('misun: here?')


                ### 6
                # # import threading, multiprocessing
                # snd, rcv = Pipe()
                # listed_args = [args]
                # # listed_args.append(callable_obj)
                # listed_args.append(snd)
                # callable_obj(listed_args)
                # # worker = threading.Thread(target=utils_inference_wrapper, args=listed_args)
                # # worker = Process(target=callable_obj, args=listed_args)
                # # worker.start()
                # # ret_val1 = rcv.recv()
                # # worker.join()
                # print('misun: here?')

                # snd, rcv = Pipe()
                # listed_args=[]
                # listed_args.append(args)
                # listed_args.append(False)
                # listed_args.append(snd)

                # # callable_obj(*listed_args)
                # # worker = threading.Thread(target=callable_obj, args=listed_args)
                # worker = Process(target=callable_obj, args=listed_args)
                # worker.start()
                # ret_val1 = rcv.recv()

                # worker.join()
                # # print('intermediate success, exit!')
                # # exit(0)

            else:
                ret_val1 = callable_obj(args)
            # try:
            #     print(f'misun: type={type(ret_val1)}, length={len(ret_val1)}')
            #     print(f'misun: type={type(ret_val1[0])}, length={len(ret_val1[0])}')
            # except:
            #     pass
            ret_val.append(ret_val1)
        elif request.num_of_returns == 2:
            ret_val1, ret_val2 = callable_obj(args)
            ret_val.append(ret_val1)
            ret_val.append(ret_val2)
        elif request.num_of_returns == 3:
            ret_val1, ret_val2, ret_val3 = callable_obj(args)
            ret_val.append(ret_val1)
            ret_val.append(ret_val2)
            ret_val.append(ret_val3)
        elif request.num_of_returns == 4:
            ret_val1, ret_val2, ret_val3, ret_val4 = callable_obj(args)
            ret_val.append(ret_val1)
            ret_val.append(ret_val2)
            ret_val.append(ret_val3)
            ret_val.append(ret_val4)
        else:
            print('error!, request.num_of_returns=',request.num_of_returns)
            return None

        response.pickled = False
        for val in ret_val:
            response.obj_ids.append(utils_set_obj(val, request.connection_id))
            
        return response

    def get_iterable_slicing(self, request, context):
        print('\nget_iterable_slcing')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.SlicingResponse()

        obj = utils_get_obj(request.iterable_id)
        sliced_obj = obj[request.start:request.end]
        response.obj_id = utils_set_obj(sliced_obj, request.connection_id)

        return response

    def config_experimental_list__physical__devices(self, request, context):
        print('\nconfig_experimental_list__physical__devices')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response=yolo_pb2.PhysicalDevices()
        
        tf_physical_devices = tf.config.experimental.list_physical_devices(request.device_type)
        for tf_physical_device in tf_physical_devices:
            physical_device = yolo_pb2.PhysicalDevices.PhysicalDevice()
            physical_device.name = tf_physical_device.name
            physical_device.device_type = tf_physical_device.device_type
            response.devices.append(physical_device)

        return response

    def image_decode__image(self, request, context):
        print('\nimage_decode__image')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response=yolo_pb2.DecodeImageResponse()
        # image_raw = tf.image.decode_image(request.byte_image, channels=request.channels)
        image_bin = open(request.image_path, 'rb').read()
        image_raw = tf.image.decode_image(image_bin, channels=request.channels, expand_animations=False)
        obj_id = utils_set_obj(image_raw, request.connection_id)
        # print(f'misun: image_raw={image_raw}, obj_id={obj_id}, shape={image_raw.shape}')

        response.obj_id=obj_id
        return response

    def expand__dims(self, request, context):
        print('\nexpand__dims')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response=yolo_pb2.ExpandDemensionResponse()
        image_obj_id = request.obj_id
        image_obj = utils_get_obj(image_obj_id)
        tensor = tf.expand_dims(image_obj, request.axis)
        tensor_obj_id = utils_set_obj(tensor, request.connection_id)
        response.obj_id=tensor_obj_id

        return response

    @ModelBuildingAPI
    def keras_Model(self, request, context):
        print('\nkeras_Model')
        _id = request.connection_id
    # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.ModelResponse()

        inputs = []
        for id in request.input_ids:
            inputs.append(utils_get_obj(id))
        outputs = []
        for id in request.output_ids:
            outputs.append(utils_get_obj(id))
        name = request.name

        result = Model(inputs, outputs, name=name)

        if request.fixed:
            if request.name not in Global_Model_Dict:
                raise IllegalModelBuildException
            else:
                # try:
                #     os.makedirs('tmp')
                # except:
                #     pass
                # # todo remove
                # tf.saved_model.save(result, 'tmp/model')
                # import json
                # json.dumps(result)
                # result.save('tmp/model.h5')
                # exit()
                # # imported = tf.saved_model.load('tmp/model', request.name)
                # imported = tf.keras.models.load_model('tmp/model.h5')
                # Global_Model_Dict[request.name].set_done(imported)
                # import json
                # json.dumps(ret_val)
                # exit()
                Global_Model_Dict[request.name].set_done(result)
        else:
            response.obj_id = utils_set_obj(result, request.connection_id)

        return response

    @ModelBuildingAPI
    def keras_layers_Input(self, request, context):
        print('\nkeras_layers_Input')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.InputResponse()
        shape=[]
        for i in request.shape:
            if i is 0:
                shape.append(None)
            else:
                shape.append(i)

        print(shape)
        inputs = Input(shape, name=request.name)

        ## because keras input is not picklable
        response.obj_id = utils_set_obj(inputs, request.connection_id)
        return response

    @ModelBuildingAPI
    def keras_layers_ZeroPadding2D(self, request, context):
        global zero_padding2d_count
        zero_padding2d_count += 1
        name = 'zero_padding2d_{:010d}'.format(zero_padding2d_count)
        request.name=name

        print('\nkeras_layers_ZeroPadding2D')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.ZeroPadding2DResponse()
        
        padding = pickle.loads(request.padding)
        data_format = None
        if len(request.data_format) > 0:
            data_format = request.data_format

        zero_padding_2d = ZeroPadding2D(padding, data_format, name=request.name)
        response.obj_id = utils_set_obj(zero_padding_2d, request.connection_id)

        return response

    @ModelBuildingAPI
    def keras_layers_Conv2D(self, request, context):
        global conv2d_count
        conv2d_count += 1
        name = 'conv2d_{:010d}'.format(conv2d_count)
        request.name=name

        print('\nkeras_layers_Conv2D')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.Conv2DResponse()

        filters = request.filters
        kernel_size = pickle.loads(request.pickled_kernel_size)
        strides = pickle.loads(request.pickled_strides)
        padding = request.padding
        use_bias = request.use_bias
        if request.pickled_kernel_regularizer is not None:
            kernel_regularizer = pickle.loads(request.pickled_kernel_regularizer)
        else:
            kernel_regularizer = None
        print('type', type(kernel_regularizer))

        conv_2d = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, kernel_regularizer=kernel_regularizer, name=request.name)

        response.obj_id = utils_set_obj(conv_2d, request.connection_id)
        return response

    @ModelBuildingAPI
    def batch_normalization(self, request, context):
        global batch_norm_count
        batch_norm_count += 1
        name = 'batchnorm_{:010d}'.format(batch_norm_count)
        request.name=name

        print('\nbatch_normalization')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.BatchNormResponse()
        callable_obj = BatchNormalization(name=request.name)

        response.obj_id = utils_set_obj(callable_obj, request.connection_id)
        return response

    @ModelBuildingAPI
    def keras_layers_LeakyReLU(self, request, context):
        global leaky_re_lu_count
        leaky_re_lu_count += 1
        name = 'leaky_re_lu_{:010d}'.format(leaky_re_lu_count)
        request.name=name

        print('\nkeras_layers_LeakyReLU')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():
        response = yolo_pb2.LeakyReluResponse()
        alpha = request.alpha

        callable_obj = LeakyReLU(alpha = alpha, name=request.name)
        print(f'leakyreluname={callable_obj.name}')
        response.obj_id = utils_set_obj(callable_obj, request.connection_id)

        return response

    @ModelBuildingAPI
    def keras_layers_Add(self, request, context):
        global add_count
        add_count += 1
        name = 'add_{:010d}'.format(add_count)
        request.name=name

        print('\nkeras_layers_Add')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.AddResponse()
        callable_obj = Add(name = request.name)
        response.obj_id = utils_set_obj(callable_obj, request.connection_id)

        return response

    def attribute_tensor_shape(self, request, context):
        print('\nattribute_tensor_shape')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.TensorShapeResponse()
        obj = utils_get_obj(request.obj_id)
        start = end = 0

        if request.start == 0:
            start = 0

        if request.end == 0:
            end = len(obj.shape)

        shape = obj.shape[start:end]
        print(shape)
        # response.pickled_shape=pickle.dumps(shape)
        # response.obj_id = utils_set_obj(shape)
        for elem in shape:
            if elem is None:
                response.shape.append(-1)
            else:
                response.shape.append(elem)

        return response

    def attribute_model_load__weights(self, request, context):
        print('\attribute_model_load__weight')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.LoadWeightsResponse()
        model_info = Global_Model_Dict[request.model_name]
        # checkpoint = model.load_weights(request.weights_path) # check if already weighted
        checkpoint = Global_Model_Dict[model_info.name].load_weights(model_info.model, request.weights_path)
        response.obj_id = utils_set_obj(checkpoint, request.connection_id)

        return response

    def attribute_checkpoint_expect__partial(self, request, context):
        print('\attribute_checkpoint_expect__partial')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.ExpectPartialResponse()
        checkpoint = utils_get_obj(request.obj_id)
        checkpoint.expect_partial()
        return response

    @ModelBuildingAPI    
    def keras_layers_Lambda(self, request, context):
        global lambda_count
        lambda_count += 1
        if request.name is None or len(request.name) is 0:
            request.name = 'lambda_{:010d}'.format(lambda_count)

        print('\nkeras_layers_Lambda')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.LambdaResponse()
        lambda_func = lambda x: eval(request.expr)
        # lambda_func = exec(request.expr)
        # print(callable(lambda_func), lambda_func.__name__) ## todo: remove
        # print(type(eval('1+2')), eval('1+2'))
        # import json
        # json_str = json.dumps(exec('1+2'))
        lambda_obj = Lambda(lambda_func, name=request.name)
        # print(callable(lambda_obj), hasattr(lambda_obj,'get_config')) ## todo: remove
        response.obj_id = utils_set_obj(lambda_obj, request.connection_id)

        return response

    @ModelBuildingAPI
    def keras_layers_UpSampling2D(self, request, context):
        print('\keras_layers_UpSampling2D')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.UpSampling2DResponse()
        callable_obj = UpSampling2D(request.size)
        response.obj_id = utils_set_obj(callable_obj, request.connection_id)

        return response

    @ModelBuildingAPI
    def keras_layers_Concatenate(self, request, context):
        print('\keras_layers_UpSampling2D')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.ContcatenateResponse()
        callable_obj = Concatenate()
        response.obj_id = utils_set_obj(callable_obj, request.connection_id)

        return response

    @ModelBuildingAPI
    def keras_regularizers_l2(self, request, context):
        print('\nkeras_regularizers_l2')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.l2Response()
        l2_value = l2(request.l)
        picked_l2 = pickle.dumps(l2_value)
        response.pickled_l2 = picked_l2

        return response

    def image_resize(self, request, context):
        print('\nimage_resize')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():
        response = yolo_pb2.ImageResizeResponse()
        image_id = request.obj_id
        image = utils_get_obj(image_id)
        size = []
        for elem in request.size:
            size.append(elem)
        
        tensor = tf.image.resize(image, size)
        response.obj_id = utils_set_obj(tensor, request.connection_id)

        return response

    def tensor_op_divide(self, request, context):
        print('\ntensor_op_divide')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.DivideResponse()
        tensor_obj = utils_get_obj(request.obj_id)
        divisor = request.divisor

        result = tensor_obj / divisor
        response.obj_id = utils_set_obj(result, request.connection_id)
        return response

    def iterable_indexing(self, request, context):
        print('\niterable_indexing')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.IndexingResponse()
        iterable = utils_get_obj(request.obj_id)
        indices = []

        ref_val = iterable[request.indices[0]]

        for index in request.indices[1:]:
            ref_val = ref_val[index]

        try:
            if len(ref_val) > 0:
                new_ref_val = [[]]
                utils_convert_elem_into_array(ref_val, new_ref_val)
                response.pickled_result = pickle.dumps(new_ref_val[0])
        except TypeError:
            response.pickled_result = pickle.dumps(ref_val.eval())

        return response

    def byte_tensor_to_numpy(self, request, context):
        print('\nbyte_tensor_to_numpy')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.TensorToNumPyResponse()

        tensor = utils_get_obj(request.obj_id)
        array = tensor.numpy()
        response.pickled_array = pickle.dumps(array)

        return response

    def get_object_by_id(self, request, context):
        print('\nget_object_by_id')
        _id = request.connection_id
        # with Global_Sess_Dict[_id].as_default(), tf.name_scope(_id), Global_Graph_Dict[_id].as_default():

        response = yolo_pb2.GetObjectResponse()

        _object = utils_get_obj(request.obj_id)
        response.object = pickle.dumps(_object)

        return response

def perf_this(pid: int):
    # -o /data/server.log
    output = subprocess.check_output(f'perf stat -p {pid} -e cycles,page-faults', shell=True, encoding='utf-8').strip()
    time.sleep(5)
    print(output)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=47), options=[('grpc.so_reuseport', 1), ('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])
    yolo_pb2_grpc.add_YoloTensorflowWrapperServicer_to_server(YoloFunctionWrapper(), server)
    server.add_insecure_port('[::]:1990')
    print('Hello TF!')
    physical_devices = tf.config.experimental.get_visible_devices('CPU')
    tf.config.threading.set_inter_op_parallelism_threads(48)
    tf.config.threading.set_intra_op_parallelism_threads(96)
    server.start()
    if True:
        process = Process(target=perf_this, args=(os.getpid(),))
        process.start()

    server.wait_for_termination()

if __name__ == '__main__':
    print('here')
    logging.basicConfig()
    FLAGS(sys.argv)
    print(f'hostroot={hostroot}')
    subprocess.check_call(f'mkdir -p {hostroot}', shell=True)
    time.sleep(3)
    serve()