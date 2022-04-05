from functools import reduce#与那个@相似，高级语法属于是

import numpy as np
from PIL import Image


def compose(*funcs):#*号见收藏夹
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)#见王毛csdn收藏夹，好复杂，功能不明
    else:
        raise ValueError('Composition of empty sequence not supported.')

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:#len(np.shape(image)) == 3:表明图像有宽，高，通道数；
        return image#np.shape(image)[2] == 3，表明有三个通道，为彩色图像
    else:
        image = image.convert('RGB')#如果不是彩色，就转换为彩色
        return image#一般不会有错误情况
    '''
    from PIL import Image 
    import numpy as np
    img = Image.open('D:\\myphoto\\mao\\2.jpg')
    a=np.shape(img)
    print(a)
    打开图像的代码
    '''
#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   获得类别
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]#Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
    return class_names, len(class_names)#是为了获得.txt文件中的类别

#---------------------------------------------------#
#   获得先验框
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''#从自己定义的文件中获取自己写的函数
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

def preprocess_input(image):
    image /= 255.0#图像归一化
    return image