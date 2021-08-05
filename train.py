import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.python.training.tracking import base

TRAIN_DIR = "./processed_data/"
IMG_SHAPE = (160,160) + (3,)
BATCH_SIZE = 16

# tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
train_datagen = ImageDataGenerator(rescale = 1/255.,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest',
validation_split=0)

train_datagen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224,224),
    class_mode="binary",
    #batch_size= BATCH_SIZE,
    shuffle=True
)




base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
include_top= False,
weights= 'imagenet')
base_model.trainable = False


model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64,'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1,activation="sigmoid")
])


base_learning_rate = 0.0001
#model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
#              loss=tf.keras.losses.BinaryCrossentropy(),
#              metrics=['accuracy'])


#history = model.fit(train_datagen,epochs= 35)


res_net50 = tf.keras.applications.ResNet50(input_shape = (224,224,3), include_top= False,weights='imagenet')
res_net50.trainable = False


res_model = tf.keras.Sequential([
    res_net50,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

res_model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss="binary_crossentropy",
              metrics=['accuracy'])

history_res = res_model.fit(train_datagen,epochs=1,batch_size= BATCH_SIZE)



res_model.evaluate(train_datagen)
