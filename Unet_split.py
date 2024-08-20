# -*- coding: utf-8 -*-
"""
@ Time:     2024/4/10 14:03 2024
@ Author:   CQshui
$ File:     Unet_split.py
$ Software: Pycharm
"""
import numpy as np
import os
os.system('chcp 65001')
import random
train_images_dir = '/datas/train/images'
train_maps_dir = '/datas/train/segmaps'
train_images = np.array(os.listdir(train_images_dir), dtype = object)
train_maps = np.array(os.listdir(train_maps_dir), dtype = object)

print(f"There are {len(train_images)} of images")
print(f"There are {len(train_maps)} of masks")

val_images_dir = 'K:\pythonProject\Torch_test\datas/val/val_images'
val_maps_dir = 'K:\pythonProject\Torch_test\datas/val/val_segmaps'

test_images_dir = 'K:\pythonProject\Torch_test\datas/test/test_images'
test_maps_dir = 'K:\pythonProject\Torch_test\datas/test/test_segmaps'

data_len = 465
def update_train():
    global train_images, train_maps
    train_images = np.array(os.listdir(train_images_dir), dtype = object)
    train_maps = np.array(os.listdir(train_maps_dir), dtype = object)

def split_data(split_img_dir, split_map_dir, percentage = 0.2):
    n = int(data_len * percentage)
    ids = np.random.choice(train_images, n, replace = False)
    for sample in ids:
        img_fp = os.path.join(train_images_dir, sample)
        map_fp = os.path.join(train_maps_dir, sample)
        os.system(f"move {img_fp} {split_img_dir}")
        os.system(f"move {map_fp} {split_map_dir}")

    #update training images list
    update_train()
split_data(val_images_dir, val_maps_dir, 0.1)
split_data(test_images_dir,test_maps_dir, 0.2)
images = np.array(os.listdir(train_images_dir), dtype = object)
maps = np.array(os.listdir(train_maps_dir), dtype = object)

print(f"There are {len(images)} of train images")
print(f"There are {len(maps)} of train masks")


val_images = np.array(os.listdir(val_images_dir), dtype = object)
val_maps = np.array(os.listdir(val_maps_dir), dtype = object)

print(f"There are {len(val_images)} of val images")
print(f"There are {len(val_maps)} of val masks")


test_images = np.array(os.listdir(test_images_dir), dtype = object)
test_maps = np.array(os.listdir(test_maps_dir), dtype = object)

print(f"There are {len(test_images)} of test images")
print(f"There are {len(test_maps)} of test masks")