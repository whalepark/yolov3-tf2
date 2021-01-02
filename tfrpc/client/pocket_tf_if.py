from enum import IntEnum, Enum
import sys, os
# self.value, self.name

CLIENT_TO_SERVER = 0x1
SERVER_TO_CLIENT = 0x2
POCKET_CLIENT = False

if os.environ.get('POCKET_CLIENT', 'False') == 'True':
    POCKET_CLIENT = True


def debug(*args):
    import inspect
    filename = inspect.stack()[1].filename
    lineno = inspect.stack()[1].lineno
    caller = inspect.stack()[1].function
    print(f'debug>> [{filename}:{lineno}, {caller}]', *args)

# if os.environ.get('POCKET_CLIENT') == 'True':
#     POCKET_CLIENT = True
#     print('POCKET_CLIENT True')
#     import yolo_msgq
#     MessageChannelInstance = yolo_msgq.PocketMessageChannel.get_instance()
# else:
#     POCKET_CLIENT = False
#     print('POCKET_CLIENT FALSE')

class PocketControl(IntEnum):
    CONNECT = 0x1
    DISCONNECT = 0x2
    HELLO = 0x3
    

class TFFunctions(IntEnum):
    LOCALQ_DEBUG = 0x00000001
    MODEL_EXIST = 0x00000002
    TF_CALLABLE = 0x00000003
    OBJECT_SLICER = 0x00000004
    TF_SHAPE = 0x00000005
    TF_RESHAPE = 0x00000006

    TF_CONFIG_EXPERIMENTAL_LIST__PHYSICAL__DEVICES = 0x10000001
    TF_CONFIG_EXPERIMENTAL_SET__MEMORY__GROWTH = 0x10000002
    TF_GRAPH_GET__TENSOR__BY__NAME = 0x10000003
    TF_KERAS_LAYERS_INPUT = 0x10000004
    TF_KERAS_LAYERS_ZEROPADDING2D = 0x10000005
    TF_KERAS_REGULARIZERS_L2 = 0x10000006
    TF_KERAS_LAYERS_CONV2D = 0x10000007
    TF_KERAS_LAYERS_BATCHNORMALIZATION = 0x10000008
    TF_KERAS_LAYERS_LEAKYRELU = 0x10000009
    TF_KERAS_LAYERS_ADD = 0x1000000a
    TF_KERAS_MODEL = 0x1000000b
    TF_KERAS_LAYERS_LAMBDA = 0x1000000c

class TFDtypes(Enum):
    tf_dtypes_float16 = 'tf.dtypes.float16'
    tf_dtypes_float32 = 'tf.dtypes.float32'
    tf_dtypes_float64 = 'tf.dtypes.float64'
    tf_dtypes_bfloat16 = 'tf.dtypes.bfloat16'
    tf_dtypes_complex64 = 'tf.dtypes.complex64'
    tf_dtypes_complex128 = 'tf.dtypes.complex128'
    tf_dtypes_int8 = 'tf.dtypes.int8'
    tf_dtypes_uint8 = 'tf.dtypes.uint8'
    tf_dtypes_uint16 = 'tf.dtypes.uint16'
    tf_dtypes_uint32 = 'tf.dtypes.uint32'
    tf_dtypes_uint64 = 'tf.dtypes.uint64'
    tf_dtypes_int16 = 'tf.dtypes.int16'
    tf_dtypes_int32 = 'tf.dtypes.int32'
    tf_dtypes_int64 = 'tf.dtypes.int64'
    tf_dtypes_bool = 'tf.dtypes.bool'
    tf_dtypes_string = 'tf.dtypes.string'
    tf_dtypes_qint8 = 'tf.dtypes.qint8'
    tf_dtypes_quint8 = 'tf.dtypes.quint8'
    tf_dtypes_qint16 = 'tf.dtypes.qint16'
    tf_dtypes_quint16 = 'tf.dtypes.quint16'
    tf_dtypes_qint32 = 'tf.dtypes.qint32'
    tf_dtypes_resource = 'tf.dtypes.resource'
    tf_dtypes_variant = 'tf.dtypes.variant'

class ReturnValue(IntEnum):
    OK = 0
    ERROR = 1
    EXCEPTIONRAISED = 2

def empty_function():
    raise Exception('Not Implemented, empty function!')

class TFDataType:
    callable_delegator = empty_function
    iterable_slicer = empty_function

    class PhysicalDevice:
        def __init__ (self, name = None, device_type = None, dict = None):
            if dict == None:
                self._typename = 'tf.config.PhysicalDevice'
                self.name = name
                self.device_type = device_type
                self.obj_id = None
            else:
                for key, value in dict.items():
                    self.__setattr__(key, value)

        def to_dict(self):
            return self.__dict__

    # class TensorShape:
    #     def __init__(self, tensor_id = None, dict = None):
    #         self._typename = 'tf.TensorShape'
    #         self.tensor_id = tensor_id
        
    #     def __getitem__(self, key):
    #         if POCKET_CLIENT is True:
    #             debug(f'key={key} id={self.tensor_id}')
    #             ret = TFDataType.iterable_slicer(self.to_dict(), key)
    #             return ret
    #         else:
    #             raise Exception('Only client can call this!')

    class Tensor:
        def __init__ (self, name = None, obj_id = None, shape=None, dict = None):
            if dict == None:
                self._typename = 'tf.Tensor'
                self.name = name
                self.obj_id = obj_id
                self.shape = shape
            else:
                debug(dict)
                for key, value in dict.items():
                    self.__setattr__(key, value)

        def to_dict(self):
            return self.__dict__
            
        def __call__(self, *args):
            if POCKET_CLIENT is True:
                ret = TFDataType.callable_delegator(self._typename, self.to_dict(), *args)
                return ret
            else:
                debug(False)

        def __getitem__(self, key):
            if POCKET_CLIENT is True:
                debug(f'key={key} name={self.name}, id={self.obj_id}')
                ret = TFDataType.iterable_slicer(self.to_dict(), key)
                return ret
            else:
                raise Exception('Only client can call this!')
        
        # def __setitem__(self, index):
        #     if POCKET_CLIENT is True:
        #         ret = TFDataType.iterable



    class Model(Tensor):
        def __init__ (self, name = None, obj_id = None, dict = None):
            if dict == None:
                self._typename = 'tf.keras.Model'
                self.name = name
                self.obj_id = obj_id
            else:
                for key, value in dict.items():
                    self.__setattr__(key, value)


    class ZeroPadding2D(Tensor):
        def __init__ (self, name = None, obj_id = None, dict = None):
            if dict == None:
                self._typename = 'tf.keras.layers.ZeroPadding2D'
                self.name = name
                self.obj_id = obj_id
            else:
                for key, value in dict.items():
                    self.__setattr__(key, value)

    class L2(Tensor):
        def __init__ (self, obj_id = None, dict = None):
            if dict == None:
                self._typename = 'tf.keras.regularizers.L2'
                self.obj_id = obj_id
            else:
                for key, value in dict.items():
                    self.__setattr__(key, value)


    class Conv2D(Tensor):
        def __init__ (self, name = None, obj_id = None, dict = None):
            if dict == None:
                self._typename = 'tf.keras.layers.Conv2D'
                self.name = name
                self.obj_id = obj_id
            else:
                for key, value in dict.items():
                    self.__setattr__(key, value)

    class BatchNormalization(Tensor):
        def __init__ (self, name = None, obj_id = None, shape=None, dict = None):
            if dict == None:
                self._typename = 'tf.keras.layers.BatchNormalization'
                self.name = name
                self.obj_id = obj_id
                self.shape = shape
            else:
                for key, value in dict.items():
                    self.__setattr__(key, value)

    class LeakyReLU(Tensor):
        def __init__ (self, name = None, obj_id = None, dict = None):
            if dict == None:
                self._typename = 'tf.keras.layers.LeakyReLU'
                self.name = name
                self.obj_id = obj_id
            else:
                for key, value in dict.items():
                    self.__setattr__(key, value)

    class Add(Tensor):
        def __init__ (self, name = None, obj_id = None, dict = None):
            if dict == None:
                self._typename = 'tf.keras.layers.Add'
                self.name = name
                self.obj_id = obj_id
            else:
                for key, value in dict.items():
                    self.__setattr__(key, value)

    class Lambda(Tensor):
        def __init__ (self, name = None, obj_id = None, dict = None):
            if dict == None:
                self._typename = 'tf.keras.layers.Add'
                self.name = name
                self.obj_id = obj_id
            else:
                for key, value in dict.items():
                    self.__setattr__(key, value)
        # def __init__ (self, function = None, output_shape=None, mask=None, arguments=None, **kwargs):
        #     self._typename = 'tf.keras.layers.Lambda'
        #     self.function = function
        #     self.output_shape = output_shape
        #     self.mask = mask
        #     self.arguments = arguments

        # def __call__ (self, input)
