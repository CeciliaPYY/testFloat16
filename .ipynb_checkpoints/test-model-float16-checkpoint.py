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
# 模型转化
net = net.cast("float16", copy = False)

t0 = time.time()
i = 0

for im in os.listdir("./images/"):
    # 数据读入
    try:
        img = mx.image.imread("./images/" + im)
	# 数据转换
	    img = img.astype("float16", copy = False)
        print("{}-{} image is processing~".format(i, im))
        i += 1
    except Exception as identifier:
        pass
    
    # 数据预处理
    transformed_img = gluoncv.data.transforms.presets.imagenet.transform_eval(img)
    transformed_img = transformed_img.astype("float16")
    # 模型推理
    net(transformed_img)
    print("OK")
    
    
# print("Ending time is: {}".format(time.time()))
t1 = time.time()
print("The Total Duration is {}".format(t1 - t0))
print("The Average Duration for one image is {}".format((t1 - t0) // i))


