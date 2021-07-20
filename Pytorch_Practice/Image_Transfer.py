# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Fri Jul  2 09:15:29 2021

@author: default
"""
import os
import pandas as pd
import numpy as np
import shutil

desk_path = os.path.join(os.path.expanduser("~"), 'Desktop')
source_folder_name = 'plant-pathology-2020-fgvc7'
source_folder_path = os.path.join(desk_path, source_folder_name)

image_label_file = os.path.join(source_folder_path, 'train.csv')
# image_label_file = os.path.join(source_folder_path, 'test.csv')
image_label_file = pd.read_csv(image_label_file)

images_path = os.path.join(source_folder_path, 'images')

image_names = image_label_file['image_id']
image_names = image_names.values.tolist()

label_names = ['healthy', 'multiple_diseases', 'rust', 'scab']
image_labels = image_label_file[label_names]
image_labels = image_labels.values.tolist()

label_indices = []
for label in image_labels:
    label_index = label.index(1)
    label_indices.append(label_index)

image_files_path = []
for image_n in image_names:
    file_path = os.path.join(images_path, image_n)
    image_files_path.append(file_path)

train_folder_path = os.path.join(source_folder_path, 'train')
val_folder_path = os.path.join(source_folder_path, 'validation')

label_indices = np.array(label_indices).reshape((-1, 1))
image_files_path = np.array(image_files_path).reshape((-1, 1))
new_arr = np.concatenate((image_files_path, label_indices), axis = 1)

for arr in new_arr:
    src = arr[0] + '.jpg'
    dst = os.path.join(train_folder_path, str(arr[1]))
    shutil.copy(src, dst)


    
    
    
    
