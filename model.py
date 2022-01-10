from  tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Dense, BatchNormalization, Activation, GlobalAveragePooling2D

from resnet_config import *

# 根据配置参数创建原版resnet或调整后的resnet
def create_model(classNums=5, flag=0):
    # 创建原版resnet
    if flag == 0:
        model = ResNet50(classNums, dropout=DROPOUT)
    # 创建调整后的resnet
    else:
        # 可以考虑将原版中的relu替换为leaky_relu，（没有进行测试）
        act = None # tf.nn.leaky_relu 
        model = MyResNet50(classNums=classNums, inputSize=INPUTSIZE, act=act, dropout=DROPOUT)
    return model

# 原版resnet50
def ResNet50(classNums, dropout=0.0):
    layers_dims=[3,4,6,3]
    
    filter_block1=[64, 64, 256]
    filter_block2=[128,128,512]
    filter_block3=[256,256,1024]
    filter_block4=[512,512,2048]

    input = Input(shape=(224,224,3))
    # stem block 
    x = Conv2D(64, (7,7), strides=(2,2),padding='same', name='stem_conv')(input)
    x = BatchNormalization(axis=3, name='stem_bn')(x)
    x = Activation('relu', name='stem_relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same', name='stem_pool')(x)
    # convolution block
    x = build_block(x, filter_block1, layers_dims[0], name='conv1')
    x = build_block(x, filter_block2, layers_dims[1], stride=2, name='conv2')
    x = build_block(x, filter_block3, layers_dims[2], stride=2, name='conv3')
    x = build_block(x, filter_block4, layers_dims[3], stride=2, name='conv4')
    # top layer
    x = GlobalAveragePooling2D(name='top_layer_pool')(x)
    if dropout:
        x = Dropout(dropout, name='dropout')(x)
    x = Dense(classNums, activation='softmax', name='fc')(x)

    model = models.Model(input, x, name='ResNet50')

    return model


# 调整后的resnet50
def MyResNet50(classNums, inputSize=(224, 224), act=None, dropout=0.0):
    
    layers_dims=[3,4,6,3]

    filter_block1=[64, 64, 256]
    filter_block2=[128,128,512]
    filter_block3=[256,256,1024]
    # 调整2：将resnet最后一个block的每个输出通道由2048减少到1024（减少了模型参数，减轻过拟合）。
    filter_block4=[512,512,1024]

    if act is None: act = 'relu'
    # 调整1：将原版中固定的模型输入尺寸，调整为支持自定义尺寸（inputSize：（w,h））
    input = Input(shape=(inputSize[1], inputSize[0], 3))
    # stem block 
    x = Conv2D(64, (7,7), strides=(2,2),padding='same', name='stem_conv')(input)
    x = BatchNormalization(axis=3, name='stem_bn')(x)
    x = Activation(act, name='stem_relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same', name='stem_pool')(x)
    # convolution block
    x = build_block(x, filter_block1, layers_dims[0], name='conv1')
    x = build_block(x, filter_block2, layers_dims[1], stride=2, name='conv2')
    x = build_block(x, filter_block3, layers_dims[2], stride=2, name='conv3')
    x = build_block(x, filter_block4, layers_dims[3], stride=2, name='conv4')
    # top layer
    x = GlobalAveragePooling2D(name='top_layer_pool')(x)
    if dropout:
        x = Dropout(dropout, name='dropout1')(x)
    # x = Dense(256, activation='softmax', name='fc1')(x)
    if dropout:
        x = Dropout(dropout, name='dropout2')(x)
    x = Dense(classNums, activation='softmax', name='fc2')(x)

    model = models.Model(input, x, name='ResNet50')

    return model

def conv_block(inputs, filter_num, stride=1, name=None, act=None):
    if act is None: act = 'relu'  
    x = inputs
    x = Conv2D(filter_num[0], (1,1), strides=stride, padding='same', name=name+'_conv1')(x)
    x = BatchNormalization(axis=3, name=name+'_bn1')(x)
    x = Activation(act, name=name+'_relu1')(x)

    x = Conv2D(filter_num[1], (3,3), strides=1, padding='same', name=name+'_conv2')(x)
    x = BatchNormalization(axis=3, name=name+'_bn2')(x)
    x = Activation(act, name=name+'_relu2')(x)

    x = Conv2D(filter_num[2], (1,1), strides=1, padding='same', name=name+'_conv3')(x)
    x = BatchNormalization(axis=3, name=name+'_bn3')(x)
    
    # residual connection
    r = Conv2D(filter_num[2], (1,1), strides=stride, padding='same', name=name+'_residual')(inputs)
    x = layers.add([x, r])
    x = Activation(act, name=name+'_relu3')(x)

    return x

def build_block (x, filter_num, blocks, stride=1, name=None, act=None):
    x = conv_block(x, filter_num, stride, name=name, act=act)
    for i in range(1, blocks):
        x = conv_block(x, filter_num, stride=1, name=name+'_block'+str(i), act=act)
    return x



if __name__ == '__main__':
    # 打印网络结构
    model = create_model(flag=1)
    model.summary()