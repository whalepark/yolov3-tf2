from enum import IntEnum
import sys, os
# self.value, self.name

CLIENT_TO_SERVER = 0x1
SERVER_TO_CLIENT = 0x2

if os.environ.get('POCKET_CLIENT') == '1':
    IS_CLIENT = True
else:
    IS_CLIENT = False

class PocketControl(IntEnum):
    CONNECT = 0x1
    DISCONNECT = 0x2
    HELLO = 0x3
    

class TFFunctions(IntEnum):
    LOCALQ_DEBUG = 0x00000001
    MODEL_EXIST = 0x00000002
    TF_CONFIG_EXPERIMENTAL_LIST__PHYSICAL__DEVICES = 0x10000001
    TF_CONFIG_EXPERIMENTAL_SET__MEMORY__GROWTH = 0x10000002
    TF_GRAPH_GET__TENSOR__BY__NAME = 0x10000003
    TF_KERAS_LAYERS_INPUT = 0x10000004
    TF_KERAS_LAYERS_ZEROPADDING2D = 0x10000005
    TF_KERAS_REGULARIZERS_L2 = 0x10000006
    TF_KERAS_LAYERS_CONV2D = 0x10000007
    TF_KERAS_LAYERS_BATCHNORMALIZATION = 0x10000008


class ReturnValue(IntEnum):
    OK = 0
    ERROR = 1
    EXCEPTIONRAISED = 2

class TFDataType:
    class PhysicalDevice:
        def __init__ (self, name, device_type, dict=None):
            if dict == None:
                self._typename = 'tf.config.PhysicalDevice'
                self.name = name
                self.device_type = device_type
                self.obj_id = None
            else:
                for key, value in dict:
                    self.__setattr__(key, value)

        def to_dict(self):
            return self.__dict__

    class Tensor:
        def __init__ (self, name, obj_id, dict=None):
            if dict == None:
                self._typename = 'tf.Tensor'
                self.name = name
                self.obj_id = obj_id
            else:
                for key, value in dict:
                    self.__setattr__(key, value)
        
        def to_dict(self):
            return self.__dict__

    class Model(Tensor):
        def __init__(self, name, obj_id, dict=None):
            if dict == None:
                self._typename = 'tf.keras.Model'
                self.name = name
                self.obj_id = obj_id
            else:
                for key, value in dict:
                    self.__setattr__(key, value)

        
        def __call__(self):
            if IS_CLIENT:
                print(True)
            else:
                print(False)

    class ZeroPadding2D(Tensor):
        def __init__(self, obj_id, dict=None):
            if dict == None:
                self._typename = 'tf.Tensor'
                self.obj_id = obj_id

        def __call__(self):
            if IS_CLIENT:
                print(True)
            else:
                print(False)

    class L2(Tensor):
        def __init__(self, obj_id, dict=None):
            if dict == None:
                self._typename = 'tf.Tensor'
                self.obj_id = obj_id

        def __call__(self):
            if IS_CLIENT:
                print(True)
            else:
                print(False)

    class Conv2D(Tensor):
        def __init__(self, obj_id, dict=None):
            if dict == None:
                self._typename = 'tf.keras.regularizers.l2'
                self.obj_id = obj_id

        def __call__(self, tensor):
            if IS_CLIENT:
                print(True)
            else:
                print(False)

    class BatchNormalization(Tensor):
        def __init__(self, obj_id, dict=None):
            if dict == None:
                self._typename = 'tf.keras.regularizers.l2'
                self.obj_id = obj_id

        def __call__(self, tensor):
            if IS_CLIENT:
                print(True)
            else:
                print(False)