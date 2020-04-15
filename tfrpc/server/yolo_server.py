#!/usr/bin/python
import os

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
from batch_norm import BatchNormalization
from utils import broadcast_iou
import threading
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
Global_Tensor_Dict = {}
Object_Ownership = {}
Connection_Set = set()

def utils_byte_chunk(data: bytes, chunk_size: int):
    start = 0
    end = chunk_size
    byte_list = []

    while True:
        if chunk_size > len(data[start:]): #last iteration
            byte_list.append(data[start:end])
            start += len(data[start:])
            end = start
            break
        else:
            byte_list.append(data[start:end])
            start += chunk_size
            end += chunk_size
    return byte_list

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
        # print my_object, 'is not iterable'
        return False      

def utils_flatten_container(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def utils_collect_garbage(connection_id: str):
    global Global_Tensor_Dict

    items_to_remove = Object_Ownership[connection_id]
    for item in items_to_remove:
        del Global_Tensor_Dict[item]

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
    def Connect(self, request, context):
        print(f'\nConnect: {request.id}')
        response = yolo_pb2.ConnectResponse()

        if request.id in Connection_Set:
            response.accept = False
        else:
            response.accept = True
            Connection_Set.add(request.id)

        return response

    def Disconnect(self, request, context):
        print(f'\nDisconnect: {request.id}')
        response = yolo_pb2.DisconnectResponse()

        Connection_Set.remove(request.id)
        threading.Thread(target = utils_collect_garbage, args=[request.id]).start()

        return response
    
    def SayHello(self, request, context):
        return yolo_pb2.HelloReply(message=f'Hello, {request.name}')

    def callable_emulator(self, request, context):
        print(f'\ncallable_emulator')
        response = yolo_pb2.CallResponse()
        
        callable_obj_id = request.callable_obj_id
        temp_args = []
        del_list = []

        ret_val = []
        ret_val1: int
        ret_val2: int
        ret_val3: int
        ret_val4: int

        if request.args_pickled:
            for arg in request.pickled_args:
                temp_args.append(pickle.loads(arg))
        else:
            for arg in request.obj_ids:
                obj = utils_get_obj(arg.obj_id)
                temp_args.append(obj)
                if arg.release:
                    del_list.append(arg.obj_id)

        if len(temp_args) > 1:
            args = temp_args
        else:
            args = temp_args[0]
        
        callable_obj = utils_get_obj(callable_obj_id)
        print(f'callable={type(callable_obj)}\narg_type={type(args)}')

        if request.num_of_returns == 1:
            ret_val1 = callable_obj(args)
            try:
                print(f'misun: type={type(ret_val1)}, length={len(ret_val1)}')
                print(f'misun: type={type(ret_val1[0])}, length={len(ret_val1[0])}')
            except:
                pass
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

        try:
            # print('try')
            # pickled_result = pickle.dumps(ret_val)
            response.pickled = True
            for elem in ret_val:
                pickled = pickle.dumps(elem)
                response.pickled_result.append(pickled)
        except TypeError:
            # print('except TypeError')
            response.pickled = False
            for val in ret_val:
                response.obj_ids.append(utils_set_obj(val, request.connection_id))
        finally:
            return response

    def get_iterable_slicing(self, request, context):
        response = yolo_pb2.SlicingResponse()

        obj = utils_get_obj(request.iterable_id)
        sliced_obj = obj[request.start:request.end]
        response.obj_id = utils_set_obj(sliced_obj, request.connection_id)

        return response

    def config_experimental_list__physical__devices(self, request, context):
        print('\nconfig_experimental_list__physical__devices')
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
        response=yolo_pb2.DecodeImageResponse()
        image_raw = tf.image.decode_image(request.byte_image, channels=request.channels)
        pickled_image = pickle.dumps(image_raw)

        response.tensor=pickled_image
        return response

    def expand__dims(self, request, context):
        print('\nexpand__dims')
        response=yolo_pb2.ExpandDemensionResponse()
        # print('misun: request.tensor=', type(request.tensor))
        unpickled_tensor = pickle.loads(request.tensor)
        # print('misun: unpickled_tensor=', type(unpickled_tensor))
        # print('misun: tensor_shape=', unpickled_tensor.shape)
        tensor = tf.expand_dims(unpickled_tensor, request.axis)
        # print('misun: tensor_type=', type(tensor), 'tensor_shape=', tensor.shape)
        pickled_tensor = pickle.dumps(tensor)
        response.tensor=pickled_tensor
        return response

    def keras_Model(self, request, context):
        print('\nkeras_Model')
        response = yolo_pb2.ModelResponse()

        inputs = []
        for id in request.input_ids:
            inputs.append(utils_get_obj(id))
        outputs = []
        for id in request.output_ids:
            outputs.append(utils_get_obj(id))
        name = request.name

        result = Model(inputs, outputs, name=name)
        response.obj_id = utils_set_obj(result, request.connection_id)
        return response


    def keras_layers_Input(self, request, context):
        print('\nkeras_layers_Input')
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

    def keras_layers_ZeroPadding2D(self, request, context):
        print('\nkeras_layers_ZeroPadding2D')
        response = yolo_pb2.ZeroPadding2DResponse()
        
        padding = pickle.loads(request.padding)
        data_format = None
        if len(request.data_format) > 0:
            data_format = request.data_format

        zero_padding_2d = ZeroPadding2D(padding, data_format, name=request.name)
        response.obj_id = utils_set_obj(zero_padding_2d, request.connection_id)

        return response

    
    def keras_layers_Conv2D(self, request, context):
        print('\nkeras_layers_Conv2D')
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

    def batch_normalization(self, request, context):
        print('\nbatch_normalization')
        response = yolo_pb2.BatchNormResponse()
        callable_obj = BatchNormalization(name=request.name)

        response.obj_id = utils_set_obj(callable_obj, request.connection_id)
        return response

    def keras_layers_LeakyReLU(self, request, context):
        print('\nkeras_layers_LeakyReLU')
        response = yolo_pb2.LeakyReluResponse()
        alpha = request.alpha

        callable_obj = LeakyReLU(alpha = alpha, name=request.name)
        response.obj_id = utils_set_obj(callable_obj, request.connection_id)

        return response

    def keras_layers_Add(self, request, context):
        print('\nkeras_layers_Add')
        response = yolo_pb2.AddResponse()
        callable_obj = Add(name = request.name)
        response.obj_id = utils_set_obj(callable_obj, request.connection_id)

        return response

    def attribute_tensor_shape(self, request, context):
        print('\nattribute_tensor_shape')

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

    def attribute_model_load__weight(self, request, context):
        print('\attribute_model_load__weight')

        response = yolo_pb2.LoadWeightsResponse()
        model = utils_get_obj(request.model_obj_id)
        checkpoint = model.load_weights(request.weights_path)
        response.obj_id = utils_set_obj(checkpoint, request.connection_id)

        return response

    def attribute_checkpoint_expect__partial(self, request, context):
        print('\attribute_checkpoint_expect__partial')

        response = yolo_pb2.ExpectPartialResponse()
        checkpoint = utils_get_obj(request.obj_id)
        checkpoint.expect_partial()
        return response
    
    def keras_layers_Lambda(self, request, context):
        print('\nkeras_layers_Lambda')

        response = yolo_pb2.LambdaResponse()
        lambda_func = lambda x: eval(request.expr)
        lambda_obj = Lambda(lambda_func, name=request.name)
        response.obj_id = utils_set_obj(lambda_obj, request.connection_id)

        return response

    def keras_layers_UpSampling2D(self, request, context):
        print('\keras_layers_UpSampling2D')

        response = yolo_pb2.UpSampling2DResponse()
        callable_obj = UpSampling2D(request.size)
        response.obj_id = utils_set_obj(callable_obj, request.connection_id)

        return response

    def keras_layers_Concatenate(self, request, context):
        print('\keras_layers_UpSampling2D')

        response = yolo_pb2.ContcatenateResponse()
        callable_obj = Concatenate()
        response.obj_id = utils_set_obj(callable_obj, request.connection_id)

        return response

    def keras_regularizers_l2(self, request, context):
        print('\nkeras_regularizers_l2')

        response = yolo_pb2.l2Response()
        l2_value = l2(request.l)
        picked_l2 = pickle.dumps(l2_value)
        response.pickled_l2 = picked_l2

        return response

    def image_resize(self, request, context):
        print('\nimage_resize')

        response = yolo_pb2.ImageResizeResponse()
        image = pickle.loads(request.pickled_image)
        size = request.size

        # print('misun: pickled_image=', type(request.pickled_image), 'image=', type(image), 'tensor_shape=', image.shape, 'image[0].shape=', image[0].shape, 'size=', request.size)
        
        tensor = tf.image.resize(image, size)
        response.pickled_tensor = pickle.dumps(tensor)

        return response

    def tensor_op_divide(self, request, context):
        print('\ntensor_op_divide')
        
        response = yolo_pb2.DivideResponse()
        tensor = pickle.loads(request.pickled_tensor)
        divisor = request.divisor

        result = tensor / divisor
        response.pickled_tensor = pickle.dumps(result)
        return response

    @staticmethod
    def iterable_indexing(self, request, contexte):
        print('\niterable_indexing')
        
        response = yolo_pb2.IndexingResponse()
        unpickled_iterable = pickle.loads(request.pickled_iterable)
        index = request.index

        try:
            for elem in unpickled_iterable[index]:
                pickled_elem = pickle.dumps(elem)
                response.elements.append(pickled_elem)
        except TypeError:
            pickled_elem = pickle.dumps(elem)
            response.elements.append(pickled_elem)

        print(f'misun: unpickled_iterable[index]={type(unpickled_iterable[index])}, len={len(unpickled_iterable[index])}')

        return response


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 1), options=[('grpc.so_reuseport', 1), ('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])
    yolo_pb2_grpc.add_YoloTensorflowWrapperServicer_to_server(YoloFunctionWrapper(), server)
    server.add_insecure_port('[::]:1990')
    print('Hello TF!')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    FLAGS(sys.argv)
    serve()