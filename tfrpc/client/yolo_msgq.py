import json
from sysv_ipc import SharedMemory, Semaphore, MessageQueue, IPC_CREX
from pocket_tf_if import TFFunctions, PocketControl, ReturnValue, CLIENT_TO_SERVER, TFDataType

def debug(*args):
    print('debug>>', *args)

class SharedMemoryChannel:
    # [0: 32) Bytes: header
    ### [0, 4) Bytes: size
    # [32, -] Bytes: data
    def __init__(self, key, size, path=None):
        self.key = key
        self.shmem = SharedMemory(key, IPC_CREX, size=size)
        self.sem = Semaphore(key, IPC_CREX, initial_value = 1)
        self.mv = memoryview(self.shmem)

        if path is not None:
            self.write(path)

    def write(self, uri, offset = 32):
        buf = open(uri, 'rb').read()
        length = len(buf)
        self.sem.acquire()
        self.mv[0:4] = length.to_bytes(4, 'little')
        self.mv[32:32+length] = buf
        # print(self.mv[32:], type(buf))
        self.sem.release()

    def read(self, size):
        self.sem.acquire()
        length = self.mv[0:4]
        data = self.mv[32:32+size]
        self.sem.release()
        return data

    def view(self, size):
        self.sem.acquire()
        self.mv = memoryview(self.shmem)
        self.sem.release()
        return self.mv[:size]

    def finalize(self):
        self.sem.remove()
        self.shmem.detach()
        self.shmem.remove()

class Utils:
    @staticmethod
    def get_container_id():
        cg = open('/proc/self/cgroup')
        content = cg.readlines()
        for line in content:
            if 'docker' in line:
                cid = line.strip().split('/')[-1]
                debug(cid)
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

    def __init__(self):
        # attach to global queue
        if PocketMessageChannel.__instance != None:
            raise Exception("Only one channel can be exist.")

        else:
            self.gq = MessageQueue(PocketMessageChannel.universal_key)
            self.conn(PocketMessageChannel.local_key)
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
        debug(json.dumps(msg, indent=2, sort_keys=True))

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
        debug(json.dumps(msg, indent=2, sort_keys=True))


    # for disconnecting
    def detach(self):
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
        debug(json.dumps(msg, indent=2, sort_keys=True))

    def check_if_model_exist(self, model_name):
        msg_type = int(TFFunctions.MODEL_EXIST)
        reply_type = msg_type | 0x40000000
        args_dict = {'model_name': model_name}
        args_dict['raw_type'] = msg_type

        args_json = json.dumps(args_dict)

        self.lq.send(args_json, block=True, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)
        
        msg = json.loads(raw_msg)
        debug(json.dumps(msg, indent=2, sort_keys=True))

        if msg['result'] == ReturnValue.OK.value:
            return msg.get('actual_return_val', None)
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_callable(self):
        pass

    def tf_config_experimental_list__physical__devices(self, device_type=None):
        msg_type = int(TFFunctions.TF_CONFIG_EXPERIMENTAL_LIST__PHYSICAL__DEVICES)
        reply_type = msg_type | 0x40000000
        args_dict = {'device_type': device_type}
        args_dict['raw_type'] = msg_type
        
        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)
        debug('tf_config_experimental_list__physical__devices', json.dumps(msg, indent=2, sort_keys=True))
        
        return msg['actual_return_val']

    def tf_config_experimental_set__memory__growth(self, device, enable):
        msg_type = int(TFFunctions.TF_CONFIG_EXPERIMENTAL_SET__MEMORY__GROWTH)
        reply_type = msg_type | 0x40000000
        args_dict = {'device': device, 'enable': enable}
        args_dict['raw_type'] = msg_type
        
        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)
        debug(json.dumps(msg, indent=2, sort_keys=True))

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
        debug(json.dumps(msg, indent=2, sort_keys=True))

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
            return msg.get('actual_return_val', None)
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
            return msg.get('actual_return_val', None)
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    def tf_keras_regularizers_l2(self, l=0.01):
        msg_type = int(TFFunctions.TF_KERAS_REGULARIZERS_L2)
        reply_type = msg_type | 0x40000000

        args_dict = {'l': l}
        args_dict['raw_type'] = msg_type

        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)
        debug(json.dumps(msg, indent=2, sort_keys=True))

        if msg['result'] == ReturnValue.OK.value:
            return msg.get('actual_return_val', None)
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

        msg_type = int(TFFunctions.TF_KERAS_LAYERS_CONV2D)
        reply_type = msg_type | 0x40000000

        args_dict = {'filters': filters, 'kernel_size': kernel_size, 'strides': strides, 'padding': padding, 'data_format': data_format, 'dilation_rate': dilation_rate, 'activation': activation, 'use_bias':use_bias, 'kernel_initializer':kernel_initializer, 
        'bias_initializer':bias_initializer,
        'kernel_regularizer':kernel_regularizer, 
        'bias_regularizer':bias_regularizer, 
        'activity_regularizer':activity_regularizer,
        'kernel_constraint':kernel_constraint, 'bias_constraint':bias_constraint}
        args_dict.update(**kwargs)
        args_dict['raw_type'] = msg_type

        args_json = json.dumps(args_dict)

        self.lq.send(args_json, type=CLIENT_TO_SERVER)
        raw_msg, _ = self.lq.receive(block=True, type=reply_type)

        msg = json.loads(raw_msg)

        if msg['result'] == ReturnValue.OK.value:
            return TFDataType.Conv2D(msg.get('actual_return_val', None))
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
            return TFDataType.BatchNormalization(msg.get('actual_return_val', None))
        elif msg['result'] == ReturnValue.EXCEPTIONRAISED.value:
            raise Exception(msg['exception'])
        else:
            raise Exception('Invalid Result!')

    # # Todo: inheritance. refer to batch_norm.py
    # def tf_keras_layers_BatchNormalization

    # def tf_keras_layers_LeakyReLU

    # def tf_keras_Model

    # def tf_keras_layers_Lambda