import os
import sys
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
        mask_path = os.path.join(self.path, id_name, "masks/")
        all_masks = os.listdir(mask_path)
        
        image = cv2.imread(image_path, 1)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        mask = np.zeros((self.image_size, self.image_size, 1))
        
        for name in all_masks:
            name_path = mask_path +  name
            name_image = cv2.imread(name_path, -1)
            name_image = cv2.resize(name_image, (self.image_size, self.image_size))
            
            name_image = np.expand_dims(name_image, axis=-1)
            mask = np.maximum(mask, name_image)
    
        image = image/255.0
        mask = mask/255.0
        
        return image, mask
    
    def __getitem__(self, index):
        if (index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
            
        file_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask = []

        for id_name in file_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
            
        image = np.array(image)
        mask = np.array(mask)
        
        return image, mask
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))
    
    
### HYPERPARAMETERS ###
IMAGE_SIZE = 128
TRAIN_PATH = "dataset/stage1_train/"
EPOCHS = 10
BATCH_SIZE = 8

train_ids = next(os.walk(TRAIN_PATH))[1]

val_data_size = 10

valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]

gen = DataGen(train_ids, TRAIN_PATH, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
x, y = gen.__getitem__(0)
print(x.shape, y.shape)


# r = random.randint(0, len(x)-1)
# fig = plt.figure()
# fig.subplots_adjust(hspace=0.4, wspace=0.4)
# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(x[r])
# ax = fig.add_subplot(1, 2, 2)
# ax.imshow(np.reshape(y[r], (IMAGE_SIZE, IMAGE_SIZE)) , cmap="gray")
# plt.show()

### DIFFERENT CONVOLUTION BLOCKS ###
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

### UNet Model ###
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

model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
model.summary()


train_gen = DataGen(train_ids, TRAIN_PATH, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
valid_gen = DataGen(valid_ids, TRAIN_PATH, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
        
train_steps = len(train_ids) // BATCH_SIZE
valid_steps = len(valid_ids) // BATCH_SIZE

model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps,
          epochs=EPOCHS)

### Save Weights ###
model.save_weights("UNetW.h5")            

x, y = valid_gen.__getitem__(1)
result = model.predict(x)

result = result > 0.5

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1,2,1)
ax.imshow(np.reshape(y[0]*255, (IMAGE_SIZE, IMAGE_SIZE)), cmap="gray")
ax = fig.add_subplot(1,2,2)
ax.imshow(np.reshape(result[0]*255, (IMAGE_SIZE, IMAGE_SIZE)), cmap="gray")
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(np.reshape(y[1]*255, (IMAGE_SIZE, IMAGE_SIZE)), cmap="gray")
ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(result[1]*255, (IMAGE_SIZE, IMAGE_SIZE)), cmap="gray")
plt.show()            
            
            
            
