import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input
from keras.models import Model

seed = 2020
random.seed = seed
np.random.seed = seed
tf.seed = seed

class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=128):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        
    def __load__(self, id_name):
        image_path = os.path.join(self.path, id_name, "images", id_name) + ".png"
        image = cv2.imread(image_path, 1)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        image = image/255.0
        return image
    
    def __getitem__(self, index):
        if (index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
            
        file_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        for id_name in file_batch:
            _img = self.__load__(id_name)
            image.append(_img)    
        image = np.array(image)
        
        return image
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))
    

def down_block(x, filters, kernel_size=(3,3), padding="same", strides=1):
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = MaxPooling2D((2,2), (2,2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3,3), padding="same", strides=1):
    us = UpSampling2D((2, 2))(x)
    concat = Concatenate()([us, skip])
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)    
    return c

def bottleneck(x, filters, kernel_size=(3,3), padding="same", strides=1):
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides)(c)
    return c

def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) # 128->64
    c2, p2 = down_block(p1, f[1]) # 64->32
    c3, p3 = down_block(p2, f[2]) # 32->16
    c4, p4 = down_block(p3, f[3]) # 16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) # 8->16
    u2 = up_block(u1, c3, f[2]) # 16->32
    u3 = up_block(u2, c2, f[1]) # 32->64
    u4 = up_block(u3, c1, f[0]) # 64->128
    
    outputs = Conv2D(1, (1,1), padding="same", activation="sigmoid")(u4)
    model = Model(inputs, outputs)
    return model

IMAGE_SIZE = 128
TEST_PATH = "dataset/stage1_test/"
BATCH_SIZE = 65

test_ids = next(os.walk(TEST_PATH))[1]

gen = DataGen(test_ids, TEST_PATH, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
x= gen.__getitem__(0)
print(x.shape)


model = UNet()
model.load_weights('UNetW.h5')

result = model.predict(x)
result = result > 0.5

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1,2,1)
ax.imshow(x[50], cmap="gray")
ax = fig.add_subplot(1,2,2)
ax.imshow(np.reshape(result[50]*255, (IMAGE_SIZE, IMAGE_SIZE)), cmap="gray")
plt.show()













