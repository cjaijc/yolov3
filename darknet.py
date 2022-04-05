from functools import wraps#使用情境：输入是函数，输出仍是函数
from keras.initializers import random_normal
'''
tf.random_normal()函数用于从“服从指定正态分布的序列”中随机取出指定个数的值。

tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
 

    shape: 输出张量的形状，必选
    mean: 正态分布的均值，默认为0
    stddev: 正态分布的标准差，默认为1.0
    dtype: 输出的类型，默认为tf.float32
    seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
    name: 操作的名称
'''
from keras.layers import (Add, BatchNormalization, Conv2D, LeakyReLU,ZeroPadding2D)
#keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None)
#这里就表示上下都补充1，所以行数就+2；左右都补充1，所以列数就+2。
'''
keras.layers.add()和keras.layer.conatenate()
add对张量执行求和运算
concatenate对张量进行串联运算
'''
from keras.regularizers import l2#正则化器
from utils.utils import compose#这里的utils库是作者写的，不是官方库。
#---------------------------------------------------#
#   单次卷积
#   DarknetCon
#
#   v2D
#---------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : random_normal(stddev=0.02)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)#python装饰器，详情见csdn博客，用户loveyyds，收藏夹“王毛”

#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):#python装饰器，详情见csdn博客，用户loveyyds，收藏夹“王毛”
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------------------------#
#   残差结构
#   首先利用ZeroPadding2D和一个步长为2x2的卷积块进行高和宽的压缩
#   然后对num_blocks进行循环，循环内部是残差结构。
#---------------------------------------------------------------------#
def resblock_body(x, num_filters, num_blocks):
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters//2, (1,1))(x)
        y = DarknetConv2D_BN_Leaky(num_filters, (3,3))(y)
        x = Add()([x,y])
    return x

#---------------------------------------------------#
#   darknet53 的主体部分
#   输入为一张416x416x3的图片
#   输出为三个有效特征层
#---------------------------------------------------#
def darknet_body(x):
    # 416,416,3 -> 416,416,32
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    # 416,416,32 -> 208,208,64
    x = resblock_body(x, 64, 1)
    # 208,208,64 -> 104,104,128
    x = resblock_body(x, 128, 2)
    # 104,104,128 -> 52,52,256
    x = resblock_body(x, 256, 8)
    feat1 = x
    # 52,52,256 -> 26,26,512
    x = resblock_body(x, 512, 8)
    feat2 = x
    # 26,26,512 -> 13,13,1024
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1, feat2, feat3

