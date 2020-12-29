import sys
import json
import tensorflow as tf

from time import sleep
from sysv_ipc import Semaphore, SharedMemory, MessageQueue, IPC_CREX, BusyError
from threading import Thread
from pocket_tf_if import PocketControl, TFFunctions, ReturnValue, TFDataType, CLIENT_TO_SERVER, SERVER_TO_CLIENT


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Make trainable=False freeze BN for real (the og version is sad)
    """

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def debug(*args):
    print('debug>>', *args)

class TensorFlowServer:
    @staticmethod
    def hello(client_id, message):
        # debug('\hello')
        return_dict = {'message': message}
        return ReturnValue.OK.value, return_dict

    @staticmethod
    def check_if_model_exist(client_id, model_name):
        keras_model = None
        if model_name in PocketManager.get_instance().model_dict:
            exist_value = True
            model = PocketManager.get_instance().model_dict[model_name]
            if model.available:
                keras_model = TFDataType.Model(model_name, id(model))
        else:
            exist_value = False

        return ReturnValue.OK.value, (exist_value, keras_model)

    @staticmethod
    def tf_config_experimental_list__physical__devices(client_id, device_type):
        # debug('\tf_config_experimental_list__physical__devices')
        device_list = tf.config.experimental.list_physical_devices(device_type)
        return_list = []
        for elem in device_list:
            return_list.append(TFDataType.PhysicalDevice(elem.__dict__))
            # return_list.append(TFDataType.PhysicalDevice(elem.name, elem.device_type).to_dict())
        return ReturnValue.OK.value, return_list

    @staticmethod
    def tf_config_experimental_set__memory__growth(client_id, device, enable):
        # debug('\tf_config_experimental_set__memory__growth')
        try:
            tf.config.experimental.set_memory_growth(device, enable)
        except Exception as e:
            import inspect
            from inspect import currentframe, getframeinfo
            frameinfo = getframeinfo(currentframe())
            return ReturnValue.EXCEPTIONRAISED.value, {'exception': e.__class__.__name__, 'message': str(e), 'filename':frameinfo.filename, 'lineno': frameinfo.lineno, 'function': inspect.stack()[0][3]}
        else:
            return ReturnValue.OK.value, []
        finally:
            pass

    # Todo: remove
    # @staticmethod
    # def tf_Graph_get__tensor__by__name(client_id, name):
    #     # debug('\ntf_Graph_get__tensor__by__name')
    #     try:
    #         pass
    #         # debug(f'name={name}, length={len(name)}')
    #         # tensor = g.get_tensor_by_name(name)
    #     except Exception as e:
    #         import inspect
    #         from inspect import currentframe, getframeinfo
    #         frameinfo = getframeinfo(currentframe())
    #         return ReturnValue.EXCEPTIONRAISED.value, {'exception': e.__class__.__name__, 'message': str(e), 'filename':frameinfo.filename, 'lineno': frameinfo.lineno, 'function': inspect.stack()[0][3]}
    #     else:
    #         return ReturnValue.OK.value, TFDataType.Tensor(tensor.name, id(tensor)).to_dict()
    #     finally:
    #         pass

    @staticmethod
    def tf_keras_layers_Input(client_id, shape=None, batch_size=None, name=None, dtype=None, sparse=False, tensor=None, ragged=False, **kwargs):
        # debug('\ntf_keras_layers_Input')
        try:
            tensor = tf.keras.layers.Input(shape=shape, batch_size=batch_size, name=name, dtype=dtype, sparse=sparse, tensor=tensor, ragged=ragged, **kwargs)
        except Exception as e:
            import inspect
            from inspect import currentframe, getframeinfo
            frameinfo = getframeinfo(currentframe())
            return ReturnValue.EXCEPTIONRAISED.value, {'exception': e.__class__.__name__, 'message': str(e), 'filename':frameinfo.filename, 'lineno': frameinfo.lineno, 'function': inspect.stack()[0][3]}
        else:
            PocketManager.get_instance().add_object_to_per_client_store(client_id, tensor)
            return ReturnValue.OK.value, TFDataType.Tensor(tensor.name, id(tensor)).to_dict()
        finally:
            pass

    @staticmethod
    def tf_keras_layers_Conv2D(client_id, filters, kernel_size, strides=(1, 1),
        padding='valid', data_format=None,
        dilation_rate=(1, 1), activation=None, use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None, **kwargs):
        # debug('\ntf_keras_layers_Conv2D')

        debug(f'kernel_regularizer={kernel_regularizer}')
        kernel_regularizer = PocketManager.get_instance().get_real_object_with_mock(client_id, kernel_regularizer)

        try:

            tensor = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)
        except Exception as e:
            import inspect
            from inspect import currentframe, getframeinfo
            frameinfo = getframeinfo(currentframe())
            return ReturnValue.EXCEPTIONRAISED.value, {'exception': e.__class__.__name__, 'message': str(e), 'filename':frameinfo.filename, 'lineno': frameinfo.lineno, 'function': inspect.stack()[0][3]}
        else:
            PocketManager.get_instance().add_object_to_per_client_store(client_id, tensor)
            return ReturnValue.OK.value, TFDataType.Tensor(tensor.name, id(tensor)).to_dict()
        finally:
            pass

    @staticmethod
    def tf_keras_layers_ZeroPadding2D(client_id, padding=(1, 1), data_format=None, **kwargs):
        # debug('\ntf_keras_layers_ZeroPadding2D')
        try:
            tensor = tf.keras.layers.ZeroPadding2D(padding=padding, data_format=data_format, **kwargs)
        except Exception as e:
            import inspect
            from inspect import currentframe, getframeinfo
            frameinfo = getframeinfo(currentframe())
            return ReturnValue.EXCEPTIONRAISED.value, {'exception': e.__class__.__name__, 'message': str(e), 'filename':frameinfo.filename, 'lineno': frameinfo.lineno, 'function': inspect.stack()[0][3]}
        else:
            PocketManager.get_instance().add_object_to_per_client_store(client_id, tensor)
            return ReturnValue.OK.value, TFDataType.Tensor(tensor.name, id(tensor)).to_dict()
        finally:
            pass

    @staticmethod
    def tf_keras_regularizers_l2(client_id, l=0.01):
        # debug('\ntf_keras_regularizers_l2')
        try:
            l2 = tf.keras.regularizers.l2(l=l)
        except Exception as e:
            import inspect
            from inspect import currentframe, getframeinfo
            frameinfo = getframeinfo(currentframe())
            return ReturnValue.EXCEPTIONRAISED.value, {'exception': e.__class__.__name__, 'message': str(e), 'filename':frameinfo.filename, 'lineno': frameinfo.lineno, 'function': inspect.stack()[0][3]}
        else:
            PocketManager.get_instance().add_object_to_per_client_store(client_id, l2)
            return ReturnValue.OK.value, TFDataType.L2(id(l2)).to_dict()
        finally:
            pass

    @staticmethod
    def tf_keras_layers_BatchNormalization(client_id, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones',
    beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
    fused=None, trainable=True, virtual_batch_size=None, adjustment=None, name=None,
    **kwargs):
        # debug('\ntf_keras_layers_BatchNormalization')
        try:
            l2 = BatchNormalization(axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale,
    beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
    moving_mean_initializer=moving_mean_initializer, moving_variance_initializer=moving_variance_initializer,
    beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
    gamma_constraint=gamma_constraint, renorm=renorm, renorm_clipping=renorm_clipping, renorm_momentum=renorm_momentum,
    fused=fused, trainable=trainable, virtual_batch_size=virtual_batch_size, adjustment=adjustment, name=name,
    **kwargs)
        except Exception as e:
            import inspect
            from inspect import currentframe, getframeinfo
            frameinfo = getframeinfo(currentframe())
            return ReturnValue.EXCEPTIONRAISED.value, {'exception': e.__class__.__name__, 'message': str(e), 'filename':frameinfo.filename, 'lineno': frameinfo.lineno, 'function': inspect.stack()[0][3]}
        else:
            PocketManager.get_instance().add_object_to_per_client_store(client_id, l2)
            return ReturnValue.OK.value, TFDataType.L2(id(l2)).to_dict()
        finally:
            pass


tf_function_dict = {
    TFFunctions.LOCALQ_DEBUG: 
    TensorFlowServer.hello,
    TFFunctions.MODEL_EXIST:
    TensorFlowServer.check_if_model_exist,
    TFFunctions.TF_CONFIG_EXPERIMENTAL_LIST__PHYSICAL__DEVICES: 
    TensorFlowServer.tf_config_experimental_list__physical__devices,
    TFFunctions.TF_CONFIG_EXPERIMENTAL_SET__MEMORY__GROWTH: 
    TensorFlowServer.tf_config_experimental_set__memory__growth,
    # TFFunctions.TF_GRAPH_GET__TENSOR__BY__NAME: 
    # TensorFlowServer.tf_Graph_get__tensor__by__name,
    TFFunctions.TF_KERAS_LAYERS_INPUT: 
    TensorFlowServer.tf_keras_layers_Input,
    TFFunctions.TF_KERAS_LAYERS_ZEROPADDING2D: 
    TensorFlowServer.tf_keras_layers_ZeroPadding2D,
    TFFunctions.TF_KERAS_REGULARIZERS_L2: 
    TensorFlowServer.tf_keras_regularizers_l2,
    TFFunctions.TF_KERAS_LAYERS_CONV2D: 
    TensorFlowServer.tf_keras_layers_Conv2D,
    TFFunctions.TF_KERAS_LAYERS_BATCHNORMALIZATION: 
    TensorFlowServer.tf_keras_layers_BatchNormalization,
}

class PocketManager:
    universal_key = 0x1001 # key for message queue
    __instance = None

    @staticmethod
    def get_instance():
        if PocketManager.__instance == None:
            PocketManager()

        return PocketManager.__instance


    def __init__(self):
        if PocketManager.__instance != None:
            raise Exception('Singleton instance exists already!')

        self.gq = MessageQueue(PocketManager.universal_key, IPC_CREX)
        self.gq_thread = Thread(target=self.pocket_new_connection)
        self.handle_clients_thread = Thread(target=self.pocket_serving_client)
        self.queues_dict = {}
        self.per_client_object_store = {}
        self.model_dict = {}
        PocketManager.__instance = self

    def start(self):
        # debug('start!')
        self.gq_thread.start()
        self.handle_clients_thread.start()

        self.handle_clients_thread.join()
        self.gq_thread.join()

    def pocket_new_connection(self):
        while True:
            raw_msg, raw_type = self.gq.receive(block=True, type=CLIENT_TO_SERVER)
            args_dict = json.loads(raw_msg)
            raw_type = args_dict['raw_type']
        
            # debug('misun>>', args_dict)
            # debug(hex(raw_type))

            type = PocketControl(raw_type)
            reply_type = raw_type | 0x40000000
            if type == PocketControl.CONNECT:
                self.add_client_queue(args_dict['client_id'], args_dict['key'])
                self.per_client_object_store[args_dict['client_id']] = {}
                self.send_ack_to_client(args_dict['client_id'])
            elif type == PocketControl.DISCONNECT:
                # Todo: Clean up
                pass
                self.per_client_object_store.pop(args_dict['client_id'], None)
            elif type == PocketControl.HELLO:
                return_dict = {'result': ReturnValue.OK.value, 'message': args_dict['message']}
                return_byte_obj = json.dumps(return_dict)
                self.gq.send(return_byte_obj, type=reply_type)
                
            sleep(0.01)

    def pocket_serving_client(self):
        while True:
            for client_id, queue in self.queues_dict.copy().items():
                try:
                    raw_msg, _ = queue.receive(block=False, type=CLIENT_TO_SERVER)

                    args_dict = json.loads(raw_msg)
                    raw_type = args_dict.pop('raw_type')
                    function_type = TFFunctions(raw_type)
                    reply_type = raw_type | 0x40000000

                    debug(f'raw_type:{hex(raw_type)}, reply_type:{hex(reply_type)}')

                    result, ret = tf_function_dict[function_type](client_id, **args_dict)
                    return_dict = {'result': result}
                    if result == ReturnValue.OK.value:
                        return_dict.update({'actual_return_val': ret})
                    else:
                        return_dict.update(ret)
                    debug(f'return_dict={return_dict}')
                    return_byte_obj = json.dumps(return_dict)
                    queue.send(return_byte_obj, type = reply_type)
                except BusyError as err:
                    pass
                sleep(0.001)
         
    def add_client_queue(self, client_id, key):
        client_queue = MessageQueue(key)
        self.queues_dict[client_id] = client_queue

    def send_ack_to_client(self, client_id):
        return_dict = {'result': ReturnValue.OK.value, 'message': 'you\'re acked!'}
        return_byte_obj = json.dumps(return_dict)
        reply_type = PocketControl.CONNECT.value | 0x40000000

        self.queues_dict[client_id].send(return_byte_obj, block=True, type=reply_type)

    def add_object_to_per_client_store(self, client_id, object):
        self.per_client_object_store[client_id][id(object)] = object

    def get_object_to_per_client_store(self, client_id, obj_id):
        return self.per_client_object_store[client_id][obj_id]

    def get_real_object_with_mock(self, client_id, mock):
        return self.per_client_object_store[client_id][mock['obj_id']]