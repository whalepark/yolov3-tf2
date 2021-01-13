from absl import flags
from absl.flags import FLAGS
import numpy as np
from .utils import broadcast_iou

import sys, os, time
cwd = os.getcwd()
os.chdir(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath('../tfrpc/client'))
from yolo_msgq import PocketMessageChannel

os.chdir(cwd)

import random

flags.DEFINE_integer('yolo_max_boxes', 100,
                     'maximum number of boxes per image')
flags.DEFINE_float('yolo_iou_threshold', 0.5, 'iou threshold')
flags.DEFINE_float('yolo_score_threshold', 0.5, 'score threshold')

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])



def debug(*args):
    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
    import inspect
    filename = inspect.stack()[1].filename
    lineno = inspect.stack()[1].lineno
    caller = inspect.stack()[1].function
    print(f'debug>> [{bcolors.WARNING}{filename}:{lineno}{bcolors.ENDC}, {caller}]', *args)


def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = PocketMessageChannel.get_instance().tf_keras_layers_ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = PocketMessageChannel.get_instance() \
                            .tf_keras_layers_Conv2D(filters=filters,
                                                    kernel_size=size,
                                                    strides=strides, 
                                                    padding=padding,
                                                    use_bias=not batch_norm, 
                                                    kernel_regularizer=PocketMessageChannel.get_instance().tf_keras_regularizers_l2(0.0005))(x)

    if batch_norm:
        x = PocketMessageChannel.get_instance().tf_keras_layers_BatchNormalization()(x)
        x = PocketMessageChannel.get_instance().tf_keras_layers_LeakyReLU(alpha=0.1)(x)
    return x

def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = PocketMessageChannel.get_instance().tf_keras_layers_Add()([prev, x])
    return x

def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x

def Darknet(name=None):
    x = inputs = PocketMessageChannel.get_instance().tf_keras_layers_Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return PocketMessageChannel.get_instance().tf_keras_Model(inputs, (x_36, x_61, x), name=name)

def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = PocketMessageChannel.get_instance().tf_keras_layers_Input(x_in[0].shape[1:]), PocketMessageChannel.get_instance().tf_keras_layers_Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = PocketMessageChannel.get_instance().tf_keras_layers_UpSampling2D(2)(x)
            x = PocketMessageChannel.get_instance().tf_keras_layers_Concatenate()([x, x_skip])
        else:
            x = inputs = PocketMessageChannel.get_instance().tf_keras_layers_Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return PocketMessageChannel.get_instance().tf_keras_Model(inputs, x, name=name)(x_in)
    return yolo_conv

def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = PocketMessageChannel.get_instance().tf_keras_layers_Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = PocketMessageChannel.get_instance().tf_keras_layers_Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)), context=locals())(x)
        return PocketMessageChannel.get_instance().tf_keras_Model(inputs, x, name=name)(x_in)
    return yolo_output

def YoloV3(size=None, channels=3, classes=80, training=False):
# anchors=yolo_anchors,
        #    masks=yolo_anchor_masks, 
    try:
        # Invalid # keras_model = tf.Graph.get_tensor_by_name('yolov3')
        is_exist, keras_model = PocketMessageChannel.get_instance().check_if_model_exist('yolov3')
    except KeyError as e:
        is_exist = False
    else:
        is_exist = True

    if is_exist:
        if keras_model == None:
            while True:
                time.sleep(random.uniform(1,3))
                keras_model = PocketMessageChannel.get_instance().check_if_model_exist('yolov3')
                if keras_model != None:
                    break
        else:
            return keras_model
    
    # x = inputs = Input([size, size, channels], name='input')
    x = inputs = PocketMessageChannel.get_instance().tf_keras_layers_Input([size, size, channels], name='input')

    # x_36, x_61, x = Darknet(name='yolo_darknet')(x)
    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(yolo_anchor_masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(yolo_anchor_masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(yolo_anchor_masks[2]), classes, name='yolo_output_2')(x)


    boxes_0 = PocketMessageChannel.get_instance().tf_keras_layers_Lambda(lambda x: yolo_boxes(x, yolo_anchors[yolo_anchor_masks[0]], classes),
                    name='yolo_boxes_0', context=locals())(output_0)


    boxes_1 = PocketMessageChannel.get_instance().tf_keras_layers_Lambda(lambda x: yolo_boxes(x, yolo_anchors[yolo_anchor_masks[1]], classes),
                     name='yolo_boxes_1', context=locals())(output_1)

    boxes_2 = PocketMessageChannel.get_instance().tf_keras_layers_Lambda(lambda x: yolo_boxes(x, yolo_anchors[yolo_anchor_masks[2]], classes),
                     name='yolo_boxes_2', context=locals())(output_2)


    outputs = PocketMessageChannel.get_instance().tf_keras_layers_Lambda(lambda x: yolo_nms(x, yolo_anchors, yolo_anchor_masks, classes),
                     name='yolo_nms', context=locals())((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return PocketMessageChannel.get_instance().tf_keras_Model(inputs, outputs, name='yolov3')
