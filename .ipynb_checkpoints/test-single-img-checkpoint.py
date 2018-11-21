import mxnet as mx
import gluoncv
import gluon
import os
import cv2
import numpy as np
import time

# 模型导入
modelname = "ResNet50_v1d"
net = gluoncv.model_zoo.get_model(modelname, pretrained = True)

# 数据读入
img = mx.image.imread("./images/terror-detect-2018-03-31T23-35-11-0fPnRaZK0B-_9Yabr-TCiw==")
# 数据预处理
transformed_img = gluoncv.data.transforms.presets.imagenet.transform_eval(img)

# 模型推理
print("Begin time is: {}".format(time.time()))
net(transformed_img)
print("Ending time is: {}".format(time.time()))

