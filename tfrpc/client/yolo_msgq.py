import json
import sys, os
# sys.path.insert(0, os.path.abspath('tfrpc/client'))
# from tfrpc.client.pocket_tf_if import POCKET_CLIENT
from types import FunctionType
from inspect import getsourcelines
from sysv_ipc import MessageQueue, IPC_CREX
from pocket_tf_if import TFFunctions, PocketControl, ReturnValue, CLIENT_TO_SERVER, TFDataType, TFDtypes, SharedMemoryChannel, ResizeMethod, POCKET_CLIENT
from time import sleep


import numpy as np

# def debug(*args):
#     class bcolors:
#         HEADER = '\033[95m'
#         OKBLUE = '\033[94m'
#         OKCYAN = '\033[96m'
#         OKGREEN = '\033[92m'
#         WARNING = '\033[93m'
#         FAIL = '\033[91m'
#         ENDC = '\033[0m'
#         BOLD = '\033[1m'
#         UNDERLINE = '\033[4m'
#     import inspect
#     filename = inspect.stack()[1].filename
#     lineno = inspect.stack()[1].lineno
#     caller = inspect.stack()[1].function
#     print(f'debug>> [{bcolors.OKCYAN}{filename}:{lineno}{bcolors.ENDC}, {caller}]', *args)

  


class Utils:
    @staticmethod
    def get_container_id():
        cg = open('/proc/self/cgroup')
        content = cg.readlines()
        for line in content:
            if 'docker' in line:
                cid = line.strip().split('/')[-1]
                # debug(cid)
                return cid
        

class PocketMessageChannel:
    universal_key = 0x1001
    client_id = Utils.get_container_id()
    local_key = int(client_id[:8], 16)
    __instance = None

    @staticmethod
    def get_instance():
        if PocketMessageChannel.__instance == None:
            PocketMessageChannel()
        
        return PocketMessageChannel.__instance

    def get_tf_callable(self):
        instance = self
        def delegate_tf_callable(*args):
            ret = instance.tf_callable(*args)
            return ret
        return delegate_tf_callable

    def get_tf_iterable_sliced(self):
        instance = self
        def delegate_tf_data_slicing(mock_dict, key):
            ret = instance.object_slicer(mock_dict, key)
            return ret
        return delegate_tf_data_slicing

    def get_model_load_weights(self):
        instance = self
        def delegate_model_load_weights(mock_dict, key):
            ret = instance.model_load_weights(mock_dict, key)
            return ret
        return delegate_model_load_weights

    def get_tensor_division(self):
        instance = self
        def delegate_tensor_division(mock_dict, other):
            ret = instance.tensor_division(mock_dict, other)
            return ret
        return delegate_tensor_division

    def disassemble_args(self, args, real_args):
        for index, elem in enumerate(args):
            real_args.append(None)
            if type(elem) in [list, tuple]:
                real_args[index] = []
                self.disassemble_args(elem, real_args[index])
            elif type(elem) is dict:
                real_args[index] = {}
                self.disassemble_kwargs(elem, real_args[index])
            else:
                if hasattr(elem, 'to_dict'):
                    real_args[index] = elem.to_dict()

    def disassemble_kwargs(self, kwargs, real_kwargs):
        for key, value in kwargs.items():
            real_kwargs[key] = None
            if type(value) in [list, tuple]:
                real_kwargs[key] = []
                self.disassemble_args(value, real_kwargs[key])
            elif type(value) is dict:
                real_kwargs[key] = {}
                self.disassemble_kwargs(value, real_kwargs[key])
            else:
                if hasattr(value, 'to_dict'):
                    real_kwargs[key] = value.to_dict()

    def __init__(self):
        # attach to global queue
        if PocketMessageChannel.__instance != None:
            raise Exception("Only one channel can be exist.")

        else:
            self.gq = MessageQueue(PocketMessageChannel.universal_key)
            self.shmem = SharedMemoryChannel(key=PocketMessageChannel.client_id, size=1 * (32 + 4 * 1024 * 1024))
            self.conn(PocketMessageChannel.local_key)
            TFDataType.callable_delegator = self.get_tf_callable()
            TFDataType.iterable_slicer = self.get_tf_iterable_sliced()
            TFDataType.Model.load_weights = self.get_model_load_weights()
            TFDataType.Tensor.tensor_division = self.get_tensor_division()
            PocketMessageChannel.__instance = self

    # control functions
    # for debugging
    def hello(self, message):
        msg_type = int(PocketControl.HELLO)
        reply_type = msg_type | 0x40000000
        args_dict = {'raw_type': msg_type, 'message': message}
        args_json = json.dumps(args_dict)

        self.gq.send(args_json, block=True, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.gq.receive(block=True, type=reply_type)
        
        msg = json.loads(raw_msg)
        # debug(json.dumps(msg, indent=2, sort_keys=True))

    # for connecting
    def conn(self, key):
        # create local queue
        self.lq = MessageQueue(key, IPC_CREX)

        msg_type = int(PocketControl.CONNECT)
        reply_type = msg_type | 0x40000000
        args_dict = {'client_id': PocketMessageChannel.client_id, 'key': key}
        args_dict['raw_type'] = msg_type
        
        args_json = json.dumps(args_dict)

        self.gq.send(args_json, type = CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)
        msg = json.loads(raw_msg)
        # debug(json.dumps(msg, indent=2, sort_keys=True))


    # for disconnecting
    def detach(self):
        msg_type = int(PocketControl.DISCONNECT)
        reply_type = msg_type | 0x40000000
        args_dict = {'client_id': PocketMessageChannel.client_id, 'key': PocketMessageChannel.local_key}
        args_dict['raw_type'] = msg_type
        args_dict['tf'] = False
        
        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type = CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)
        self.lq.remove()
        msg = json.loads(raw_msg)
        pass
        # self.mq.send('detach')
        # self.mq.remove()

    # for debugging    
    def hello_via_lq(self, message):
        msg_type = int(TFFunctions.LOCALQ_DEBUG)
        reply_type = msg_type | 0x40000000
        args_dict = {'message': message}
        args_dict['raw_type'] = msg_type
        
        args_json = json.dumps(args_dict)

        self.lq.send(args_json, block=True, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)
        # debug(json.dumps(msg, indent=2, sort_keys=True))

    def check_if_model_exist(self, model_name):
        msg_type = int(TFFunctions.MODEL_EXIST)
        reply_type = msg_type | 0x40000000
        args_dict = {'model_name': model_name}
        args_dict['raw_type'] = msg_type

        args_json = json.dumps(args_dict)

        self.lq.send(args_json, block=True, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)
        
        msg = json.loads(raw_msg)
        # debug(json.dumps(msg, indent=2, sort_keys=True))

        if msg['result'] == ReturnValue.OK.value:
            ret = msg.get('actual_return_val', None)
            if ret[1] is not None:
                ret[1] = TFDataType.Model(dict=ret[1])
                return ret
            else:
                return ret
            # return msg.get('actual_return_val', None)
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_callable(self, typename, callable, *args):
        msg_type = int(TFFunctions.TF_CALLABLE)
        reply_type = msg_type | 0x40000000
        args_dict = {'typename': typename, 'callable': callable, 'args': args}
        args_dict['raw_type'] = msg_type
        args_list = list(args)
        args_dict['args'] = args_list

        self.convert_object_to_dict(old_list=args_list)

        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)
        # debug(raw_msg)

        msg = json.loads(raw_msg)
        if msg['result'] == ReturnValue.OK.value:
            # return TFDataType.Tensor(dict=msg['actual_return_val'])
            ret = msg['actual_return_val']
            # debug(f'type(ret)={type(ret)}')
            if type(ret) is list:
                ret_list = [TFDataType.Tensor(dict=item) for item in ret]
                # [debug(item.obj_id) for item in ret_list]
                return ret_list
            else:
                ret = TFDataType.Tensor(dict=ret)
                # debug(ret.__dict__)
                return ret
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def object_slicer(self, mock_dict, key):
        msg_type = int(TFFunctions.OBJECT_SLICER)
        reply_type = msg_type | 0x40000000
        args_dict = {'mock_dict': mock_dict, 'key': key}
        args_dict['raw_type'] = msg_type

        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)
        # debug('tf_callable', json.dumps(msg, indent=2, sort_keys=True))

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.Tensor.make_tensor(dict=msg['actual_return_val'])
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tensor_division(self, mock_dict, other):
        msg_type = int(TFFunctions.TENSOR_DIVISION)
        reply_type = msg_type | 0x40000000
        args_dict = {'mock_dict': mock_dict, 'other': other}
        args_dict['raw_type'] = msg_type

        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)
        # debug('tf_callable', json.dumps(msg, indent=2, sort_keys=True))

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.Tensor.make_tensor(dict=msg['actual_return_val'])
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def model_load_weights(self, model, filepath, by_name=False, skip_mismatch=False):
        msg_type = int(TFFunctions.TF_MODEL_LOAD_WEIGHTS)
        reply_type = msg_type | 0x40000000
        args_dict = {'model': model, 'filepath': filepath, 'by_name': by_name, 'skip_mismatch': skip_mismatch}
        args_dict['raw_type'] = msg_type

        self.convert_object_to_dict(args_dict)
        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)
        # debug('tf_callable', json.dumps(msg, indent=2, sort_keys=True))

        if msg['result'] == ReturnValue.OK.value:
            return
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_shape(self, input, out_type=TFDtypes.tf_dtypes_int32, name=None):
        msg_type = int(TFFunctions.TF_SHAPE)
        reply_type = msg_type | 0x40000000

        args_dict = {'input': input, 'out_type': out_type, 'name': name}
        args_dict['raw_type'] = msg_type

        self.convert_object_to_dict(args_dict)
        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.Tensor(dict=msg.get('actual_return_val', None))
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_reshape(self, tensor, shape, name=None):
        msg_type = int(TFFunctions.TF_RESHAPE)
        reply_type = msg_type | 0x40000000

        args_dict = {'tensor': tensor, 'shape': shape, 'name': name}
        # args_dict.update(**kwargs)
        args_dict['raw_type'] = msg_type

        self.convert_object_to_dict(args_dict)
        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.Tensor(dict=msg.get('actual_return_val', None))
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_config_experimental_list__physical__devices(self, device_type=None):
        msg_type = int(TFFunctions.TF_CONFIG_EXPERIMENTAL_LIST__PHYSICAL__DEVICES)
        reply_type = msg_type | 0x40000000
        args_dict = {'device_type': device_type}
        args_dict['raw_type'] = msg_type
        
        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)
        # debug('tf_config_experimental_list__physical__devices', json.dumps(msg, indent=2, sort_keys=True))
        
        if msg['result'] == ReturnValue.OK.value:
            return [TFDataType.PhysicalDevice(dict=item) for item in msg['actual_return_val']]
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_config_experimental_set__memory__growth(self, device, enable):
        msg_type = int(TFFunctions.TF_CONFIG_EXPERIMENTAL_SET__MEMORY__GROWTH)
        reply_type = msg_type | 0x40000000
        args_dict = {'device': device, 'enable': enable}
        args_dict['raw_type'] = msg_type
        
        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)
        # debug(json.dumps(msg, indent=2, sort_keys=True))

        if msg['result'] == ReturnValue.OK.value:
            return msg['actual_return_val']
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_Graph_get__tensor__by__name(self, name):
        msg_type = int(TFFunctions.TF_GRAPH_GET__TENSOR__BY__NAME)
        reply_type = msg_type | 0x40000000
        args_dict = {'name': name}
        args_dict['raw_type'] = msg_type

        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)
        # debug(json.dumps(msg, indent=2, sort_keys=True))

        if msg['result'] == ReturnValue.OK.value:
            return msg.get('actual_return_val', None)
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')
        


    def tf_keras_layers_Input(self, shape=None, batch_size=None, name=None, dtype=None, sparse=False, tensor=None, ragged=False, **kwargs):
        msg_type = int(TFFunctions.TF_KERAS_LAYERS_INPUT)
        reply_type = msg_type | 0x40000000

        args_dict = {'shape': shape, 'batch_size': batch_size, 'name': name, 'dtype': dtype, 'sparse': sparse, 'tensor': tensor, 'ragged': ragged}
        args_dict.update(**kwargs)
        args_dict['raw_type'] = msg_type

        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)

        if msg['result'] == ReturnValue.OK.value:
            ret = TFDataType.Tensor(dict=msg.get('actual_return_val', None))
            return ret
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')


    def tf_keras_layers_ZeroPadding2D(self, padding=(1, 1), data_format=None, **kwargs):
        msg_type = int(TFFunctions.TF_KERAS_LAYERS_ZEROPADDING2D)
        reply_type = msg_type | 0x40000000

        args_dict = {'padding': padding, 'data_format': data_format}
        args_dict.update(**kwargs)
        args_dict['raw_type'] = msg_type

        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.ZeroPadding2D(dict=msg.get('actual_return_val', None))
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_keras_regularizers_l2(self, l=0.01):
        msg_type = int(TFFunctions.TF_KERAS_REGULARIZERS_L2) ###
        reply_type = msg_type | 0x40000000

        args_dict = {'l': l} ###
        args_dict['raw_type'] = msg_type

        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)
        # debug(json.dumps(msg, indent=2, sort_keys=True))

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.L2(dict=msg.get('actual_return_val', None)) ###
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_keras_layers_Conv2D(self, filters, kernel_size, strides=(1, 1),
        padding='valid', data_format=None,
        dilation_rate=(1, 1), activation=None, use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None, **kwargs):

        msg_type = int(TFFunctions.TF_KERAS_LAYERS_CONV2D) ###
        reply_type = msg_type | 0x40000000

        args_dict = {'filters': filters, 'kernel_size': kernel_size, 'strides': strides, 'padding': padding, 'data_format': data_format, 'dilation_rate': dilation_rate, 'activation': activation, 'use_bias':use_bias, 'kernel_initializer':kernel_initializer, 
        'bias_initializer':bias_initializer,
        'kernel_regularizer':kernel_regularizer, 
        'bias_regularizer':bias_regularizer, 
        'activity_regularizer':activity_regularizer,
        'kernel_constraint':kernel_constraint, 'bias_constraint':bias_constraint} ###
        args_dict.update(**kwargs)
        args_dict['raw_type'] = msg_type
        for key, value in args_dict.copy().items():
            if hasattr(value, 'to_dict'):
                args_dict[key] = value.to_dict()

        args_json = json.dumps(args_dict)
        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.Conv2D(dict=msg.get('actual_return_val', None)) ###
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_keras_layers_BatchNormalization(self,
    axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones',
    beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
    fused=None, trainable=True, virtual_batch_size=None, adjustment=None, name=None,
    **kwargs):
        msg_type = int(TFFunctions.TF_KERAS_LAYERS_BATCHNORMALIZATION)
        reply_type = msg_type | 0x40000000

        args_dict = {'axis': axis, 'momentum': momentum, 'epsilon': epsilon, 'center': center, 'scale': scale, 'beta_initializer': beta_initializer, 'gamma_initializer': gamma_initializer, 'moving_mean_initializer':moving_mean_initializer, 'moving_variance_initializer':moving_variance_initializer, 
        'beta_regularizer':beta_regularizer,
        'gamma_regularizer':gamma_regularizer, 
        'beta_constraint':beta_constraint, 
        'gamma_constraint':gamma_constraint,
        'renorm':renorm, 'renorm_clipping':renorm_clipping, 'renorm_momentum': renorm_momentum, 'fused': fused, 'trainable': trainable, 'virtual_batch_size': virtual_batch_size, 'adjustment': adjustment, 'name': name}
        args_dict.update(**kwargs)
        args_dict['raw_type'] = msg_type

        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.BatchNormalization(dict=msg.get('actual_return_val', None))
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_keras_layers_LeakyReLU(self, alpha=0.3, **kwargs):
        msg_type = int(TFFunctions.TF_KERAS_LAYERS_LEAKYRELU)
        reply_type = msg_type | 0x40000000

        args_dict = {'alpha': alpha}
        args_dict.update(**kwargs)
        args_dict['raw_type'] = msg_type

        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.LeakyReLU(dict=msg.get('actual_return_val', None))
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_keras_layers_Add(self, **kwargs):
        msg_type = int(TFFunctions.TF_KERAS_LAYERS_ADD) ###
        reply_type = msg_type | 0x40000000

        args_dict = {} ###
        args_dict.update(**kwargs)
        args_dict['raw_type'] = msg_type
        for key, value in args_dict.copy().items():
            if hasattr(value, 'to_dict'):
                args_dict[key] = value.to_dict()

        args_json = json.dumps(args_dict)
        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.Add(dict=msg.get('actual_return_val', None)) ###
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_keras_Model(self, *args, **kwargs):
        msg_type = int(TFFunctions.TF_KERAS_MODEL) ###
        reply_type = msg_type | 0x40000000

        args_dict = {'args': args} ###
        args_dict.update(**kwargs)
        args_dict['raw_type'] = msg_type

        self.convert_object_to_dict(args_dict)

        args_json = json.dumps(args_dict)
        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.Model(dict=msg.get('actual_return_val', None)) ###
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_keras_layers_Lambda(self, function, output_shape=None, mask=None, arguments=None, **kwargs):
        msg_type = int(TFFunctions.TF_KERAS_LAYERS_LAMBDA) ###
        reply_type = msg_type | 0x40000000

        function, raw_args = self.__get_str_from_lambda(function)
        args_dict = {'function': function, 'output_shape': output_shape, 'mask': mask, 'arguments': arguments} ###
        args_dict.update(**kwargs)
        args_dict['raw_type'] = msg_type

        context = kwargs.get('context', None)
        if context is None:
            raise Exception('context info is needed!')
        context = self.__context_filter(context, function, raw_args)
        function = self.__substitute_closure_vars_with_context(function, context)
        args_dict.pop('context')
        args_dict['function'] = function

        self.convert_object_to_dict(args_dict)

        args_json = json.dumps(args_dict)
        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.Lambda(dict=msg.get('actual_return_val', None)) ###
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_keras_layers_UpSampling2D(self, size=(2, 2), data_format=None, interpolation='nearest', **kwargs):
        msg_type = int(TFFunctions.TF_KERAS_LAYERS_UPSAMPLING2D) ###
        reply_type = msg_type | 0x40000000

        args_dict = {'size':size, 'data_format':data_format} ###
        args_dict.update(**kwargs)
        args_dict['raw_type'] = msg_type
        for key, value in args_dict.copy().items():
            if hasattr(value, 'to_dict'):
                args_dict[key] = value.to_dict()

        args_json = json.dumps(args_dict)
        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.Add(dict=msg.get('actual_return_val', None)) ###
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_keras_layers_Concatenate(self, axis=-1, **kwargs):
        msg_type = int(TFFunctions.TF_KERAS_LAYERS_CONCATENATE) ###
        reply_type = msg_type | 0x40000000

        args_dict = {'axis':axis} ###
        args_dict.update(**kwargs)
        args_dict['raw_type'] = msg_type
        for key, value in args_dict.copy().items():
            if hasattr(value, 'to_dict'):
                args_dict[key] = value.to_dict()

        args_json = json.dumps(args_dict)
        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.Add(dict=msg.get('actual_return_val', None)) ###
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_image_decode__image(self, contents, channels=None, dtype=TFDtypes.tf_dtypes_uint8, name=None, expand_animations=True):
        msg_type = int(TFFunctions.TF_IMAGE_DECODE__IMAGE) ###
        reply_type = msg_type | 0x40000000

        args_dict = {'contents': contents, 'channels': channels, 'dtype': dtype, 'name': name, 'expand_animations': expand_animations} ###
        args_dict['raw_type'] = msg_type
        self.convert_object_to_dict(args_dict)

        args_json = json.dumps(args_dict)
        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.Tensor(dict=msg.get('actual_return_val', None)) ###
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_expand__dims(self, input, axis, name=None):
        msg_type = int(TFFunctions.TF_EXPAND__DIMS) ###
        reply_type = msg_type | 0x40000000

        args_dict = {'input': input, 'axis': axis, 'name': name} ###
        args_dict['raw_type'] = msg_type
        self.convert_object_to_dict(args_dict)

        args_json = json.dumps(args_dict)
        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.Tensor(dict=msg.get('actual_return_val', None)) ###
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_image_resize(self, images, size, method=ResizeMethod.BILINEAR, preserve_aspect_ratio=False,
    antialias=False, name=None):
        msg_type = int(TFFunctions.TF_IMAGE_RESIZE) ###
        reply_type = msg_type | 0x40000000

        if type(method) is not str:
            method = method.value

        args_dict = {'images': images, 'size': size, 'method': method, 'preserve_aspect_ratio': preserve_aspect_ratio, 'antialias': antialias, 'name': name} ###
        args_dict['raw_type'] = msg_type
        self.convert_object_to_dict(args_dict)

        args_json = json.dumps(args_dict)
        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.Tensor(dict=msg.get('actual_return_val', None)) ###
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def __remove_code_after_lambda(self, string):
        parenthese_cursor = 1
        new_substring = ''
        for index in range(0, len(string)):
            if string[index] == '(':
                parenthese_cursor += 1
                # debug(parenthese_cursor, index, string[index+1:])
            elif string[index] == ')':
                parenthese_cursor -= 1
                # debug(parenthese_cursor, index, string[index+1:])
            elif parenthese_cursor == 1 and string[index] == ',':
                parenthese_cursor -= 1

            if parenthese_cursor is 0:
                new_substring = 'lambda' + string[0:index]
                break
        return new_substring

    def __get_str_from_lambda(self, func):
        func_string = str(getsourcelines(func)[0])
        func_string = func_string.split('lambda', 2)[1]
        raw_args = [elem.strip() for elem in func_string.split(':')[0].split(',')]
        func_string = self.__remove_code_after_lambda(func_string)
        return func_string, raw_args

    def __context_filter(self, context, function, function_param):
        new_context = {}
        for key, value in context.items():
            if key in function and key not in function_param:
                new_context[key] = value

        return new_context

    def __substitute_closure_vars_with_context(self, function, context):
        new_string = function
        for key, value in context.copy().items():
            index = 0
            while index < len(function):
                if function[index:].startswith(key) and \
                   not function[index-1].isalnum() and \
                   not function[index+len(key)].isalnum():
                   substitute = str(value)
                   new_string = function[:index] + function[index:].replace(key, substitute, 1)
                   function = new_string
                index += 1
            function = new_string
        return function



    def convert_object_to_dict(self, old_dict: dict = None, old_list: list = None):
        if old_dict is not None:
            for key, value in old_dict.copy().items():
                datatype = type(value)
                if isinstance(value, TFDataType.Tensor):
                    old_dict[key] = value.to_dict()
                elif datatype is TFDtypes:
                    old_dict[key] = value.value
                elif datatype is list:
                    self.convert_object_to_dict(old_list = value)
                elif datatype is tuple:
                    old_dict[key] = tuple_to_list = list(value)
                    self.convert_object_to_dict(old_list = tuple_to_list)
                elif datatype is dict and 'to_dict' not in value:
                    self.convert_object_to_dict(old_dict = value)
                elif datatype in (int, float, bool, str, bytearray, type(None)):
                    pass
                elif datatype is FunctionType:
                    lambda_str = self.__get_str_from_lambda(value)
                    old_dict[key] = lambda_str
                    # debug(lambda_str)
                elif datatype is np.ndarray:
                    old_dict[key] = value.tolist()
                    pass
                elif datatype is bytes:
                    # old_dict[key] = value.decode()
                    # debug(type(old_dict[key]))
                    old_dict[key] = len(value)
                    self.shmem.write(contents=value)
                else:
                    old_dict[key] = value.__dict__
                    # debug(f'[warning] unknown type {type(value)} is converted to dict.')
                    # raise Exception(f'No such type! Error! {type(value)}')

        if old_list is not None:
            for index, elem in enumerate(old_list):
                datatype = type(elem)
                if isinstance(elem, TFDataType.Tensor):
                    old_list[index] = elem.to_dict()
                elif datatype is TFDtypes:
                    old_list[index] = elem.value
                elif datatype is list:
                    self.convert_object_to_dict(old_list = elem)
                elif datatype is tuple:
                    old_list[index] = tuple_to_list = list(elem)
                    self.convert_object_to_dict(old_list = tuple_to_list)
                elif datatype is dict and 'to_dict' not in elem:
                    self.convert_object_to_dict(old_dict = elem)
                elif datatype in (int, float, bool, str,bytearray, type(None)):
                    pass
                elif datatype is FunctionType:
                    lambda_str = self.__get_str_from_lambda(elem)
                    old_list[index] = lambda_str
                    # debug(lambda_str)
                elif datatype is np.ndarray:
                    old_list[index] = elem.tolist()
                    pass
                elif datatype is bytes:
                    old_list[index] = len(elem)
                    self.shmem.write(contents=elem)
                else:
                    old_list[index] = elem.__dict__
                    # debug(f'[warning] unknown type {type(elem)} is converted to dict.')
                    # raise Exception(f'No such type! Error! {type(elem)}')
