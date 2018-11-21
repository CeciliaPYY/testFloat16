import pandas as pd
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
import gluoncv
import cv2
from mxnet import ndarray as nd
import numpy as np
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.block import Block, HybridBlock


class BKData(ImageFolderDataset):
    def __init__(self, *arg1, **arg2):
        super(BKData, self).__init__(*arg1,**arg2)

    def __getitem__(self, idx):
        '''
        use cv2 backend
        '''
        img  = cv2.imread(self.items[idx][0])
        img  = nd.array( img[:,:,:3]).astype(np.uint8)
#        img = img.astype('float16', copy=False)
        label = self.items[idx][1]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.Cast()
])

test_path = "../classified-images"
batch_size = 256
num_gpus = 1
num_workers = 8
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]


test_data = gluon.data.DataLoader(
    BKData(test_path).transform_first(transform_test),
    batch_size=batch_size, shuffle=True, num_workers = num_workers,last_batch = 'discard')

from mxnet.gluon import nn
from mxnet import gluon, image, init, nd
model_name = 'ResNet50_v1d'
finetune_net = gluoncv.model_zoo.get_model(model_name, pretrained=True)

with finetune_net.name_scope():
    finetune_net.fc = nn.Dense(2)
finetune_net.fc.initialize(init.Xavier(), ctx = ctx)
finetune_net.collect_params().reset_ctx(ctx)
finetune_net.hybridize()
finetune_net.cast('float16')

# 转换后测试
import time
t1 = time.time()
ctx  = [mx.gpu(0)]

for idx , g in enumerate(test_data):
    g[0] = g[0].as_in_context(mx.gpu(0))
    y = finetune_net(g[0])

print("After Transform: Total Duration is {}".format(time.time() - t1))
print("After Transform: Average Duration is {}".format((time.time() - t1) / (len(test_data)*batch_size)))
