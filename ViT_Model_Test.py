# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Sun May  2 16:29:21 2021

@author: Admin
"""
import os
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from learning_rate import choose
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ViT_model import VisionTransformer

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Transformer Hyper-parameters
image_size = 256
patch_size = 32
num_layers = 4
d_model = 64
num_heads = 8
mlp_dim = 128
dropout = 0.1
num_classes = 25

# model training hyper-parameters
learning_rate = 0.001
batch_size = 32
epochs = 500

train_data_dir = os.path.join(os.getcwd(), 'tiny_test/train')
validation_data_dir = os.path.join(os.getcwd(), 'tiny_test/validation')

nb_train_samples = 10000
nb_validation_samples = 2500


model = VisionTransformer(image_size = image_size, 
                          patch_size = patch_size, 
                          num_layers = num_layers, 
                          num_classes = num_classes, 
                          d_model = d_model, 
                          num_heads = num_heads, 
                          mlp_dim = mlp_dim, 
                          channels = 3, 
                          dropout = dropout)

optimizer = SGD(learning_rate = learning_rate, momentum = 0.9, nesterov = True)
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              optimizer = optimizer, 
              metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rotation_range = 15, 
                                   width_shift_range = 0.1, 
                                   height_shift_range = 0.1,
                                   brightness_range = [0.5, 2.0], 
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='sparse')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='sparse')

# callbacks defination
lr_monitorable = True
lr_reduce = choose(lr_monitorable = lr_monitorable)
callbacks = [lr_reduce]

#model fit
hist = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size, 
    callbacks=callbacks)

#print acc and stored into acc.txt
f = open('/home/xingshuli/Desktop/acc.txt','w')
f.write(str(hist.history['acc']))
f.close()
#print val_acc and stored into val_acc.txt
f = open('/home/xingshuli/Desktop/val_acc.txt','w')
f.write(str(hist.history['val_acc']))
f.close()
#print val_loss and stored into val_loss.txt   
f = open('/home/xingshuli/Desktop/val_loss.txt', 'w')
f.write(str(hist.history['val_loss']))
f.close()

#the reasonable accuracy of model should be calculated based on
#the value of patience in EarlyStopping: accur = accur[-patience + 1:]/patience
Er_patience = 10  # Er_patience = patience + 1
accur = []
with open('/home/xingshuli/Desktop/val_acc.txt','r') as f1:
    data1 = f1.readlines()
    for line in data1:
        odom = line.strip('[]\n').split(',')
        num_float = list(map(float, odom))
        accur.append(num_float)
f1.close()

y = sum(accur, [])
ave = sum(y[-Er_patience:]) / len(y[-Er_patience:])
print('Validation Accuracy = %.4f' % (ave))



