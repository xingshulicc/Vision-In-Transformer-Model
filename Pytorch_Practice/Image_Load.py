# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Fri Jul 23 17:33:00 2021

@author: default
"""
'''
how to use ImageFolder function to load images and labels

Parameters:
    root: string, Root directory path.
    transform:  A function/transform that takes in an PIL image and returns a transformed version.
    target_transform: A function/transform that takes in the target (Image Label) and transforms it.
    loader: A function to load an image given its path, the default image format is PIL
    is_valid_file: A function that takes path of an Image file and check if the file is a valid file.
    

'''


import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os

# step 1: data augmentation

image_height = 224
image_width = 224

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

batch_size = 16


root = os.path.join(os.getcwd(),'data')
train_dic = os.path.join(root, 'train')
val_dic = os.path.join(root, 'validation')

transform_train = transforms.Compose([
    transforms.Resize(size=(image_height, image_width)), 
    transforms.RandomAffine(
        degrees=30, 
        translate=(0.1, 0.1), 
        scale=(0.9, 1.1), 
        fillcolor=0, 
        ), 
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.2, 
        ), 
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(p=0.5), 
    transforms.ToTensor(), 
    transforms.Normalize(norm_mean, norm_std), 
    ])

transform_val = transforms.Compose([
    transforms.Resize(size=(image_height, image_width)), 
    transforms.ToTensor(), 
    transforms.Normalize(norm_mean, norm_std), 
    ])

# step 2: load image and assign image label

train_dataset = ImageFolder(
    root = train_dic, 
    transform = transform_train, 
    )

val_dataset = ImageFolder(
    root = val_dic, 
    transform = transform_val, 
    )

# step 3: create a Dataloader 

train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset, 
    batch_size = batch_size, 
    shuffle = True, 
    num_workers = 2, 
    pin_memory = False, 
    drop_last = False, 
    )

val_loader = torch.utils.data.DataLoader(
    dataset = val_dataset, 
    batch_size = batch_size, 
    shuffle = False, 
    num_workers = 2, 
    pin_memory = False, 
    drop_last = False, 
    )


# print('*********************')
# print(train_dataset.class_to_idx)
# print('*********************')
# print(train_dataset.imgs)
# print('*********************')
# print(train_dataset.targets)

