# -*- coding: utf-8 -*-
"""
@ Time:     2024/4/24 23:41 2024
@ Author:   CQshui
$ File:     Unet_test.py
$ Software: Pycharm
"""
import cv2
from PIL import Image
import albumentations as A
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import Sequence
import segmentation_models as sm
import tensorflow as tf


class Dataset:
    def __init__(self,
                 images_dir,
                 maps_dir,
                 augmentation=None,
                 preprocessing=None):

        self.img_ids = os.listdir(images_dir)

        self.img_fps = [os.path.join(images_dir, id) for id in self.img_ids]
        self.map_fps = [os.path.join(maps_dir, id) for id in self.img_ids]

        self.maps_dir = maps_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_fps[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))

        map = Image.open(os.path.join(self.map_fps[idx]))
        map = np.array(map).astype('uint8')
        map[map > 0] = 1
        map = cv2.resize(map, (512, 512))
        map = np.expand_dims(map, axis=-1)

        if self.augmentation:
            aug_sample = self.augmentation(image=image, mask=map)
            image, map = aug_sample['image'], aug_sample['mask']

        if self.preprocessing:
            processed_sample = self.preprocessing(image=image, mask=map)
            image, map = processed_sample['image'], processed_sample['mask']

        return image, map.astype(np.float32)

    def __len__(self):
        return len(self.img_ids)


class DataLoader(Sequence):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.indices = np.arange(len(dataset))
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size

        data = []
        for idx in range(start, stop):
            data.append(self.dataset[idx])

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        return len(self.indices) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indices = np.random.permutation(self.indices)


def get_preprocessing(preprocessing_fn):
    _transform = [
            A.Lambda(image=preprocessing_fn),
        ]
    return A.Compose(_transform)


predict_images_dir = r'I:\0_Datas\Working\gray'
predict_maps_dir = r'I:\0_Datas\Working\labeled'

sm.set_framework('tf.keras')
BACKBONE = 'efficientnetb3'
ACTIVATION = 'sigmoid'
BATCH_SIZE = 8
NUM_CLASSES = 1
LR = 1e-4
EPOCHS = 1


preprocess_input = sm.get_preprocessing(BACKBONE)
model = sm.Unet(BACKBONE, classes=NUM_CLASSES, activation=ACTIVATION)

optim = tf.keras.optimizers.Adam(LR)
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

model.compile(optim, total_loss, metrics)


test_dataset = Dataset(
    predict_images_dir,
    predict_maps_dir,
    preprocessing=get_preprocessing(preprocess_input),
)


test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
model.load_weights('best_model.h5')
scores = model.evaluate_generator(test_dataloader)

print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))

# 画出示例图
n = 5
ids = np.random.choice(np.arange(len(test_dataset)), size=n, replace=False)

fig, axes = plt.subplots(n, 3, figsize = (16,25))

cols = ['Image', "Ground Truth", "Prediction"]
for ax, col in zip(axes[0], cols):
    ax.set_title(col)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

axes = axes.ravel()
for i,id in enumerate(ids):
  image, gt_mask = test_dataset[id]
  image = np.expand_dims(image, axis=0)
  pr_mask = model.predict(image).round()

  # 由于用cv2读取，plt输出
  image_plt = cv2.cvtColor(np.squeeze(image),cv2.COLOR_RGB2GRAY)

  for j, im in enumerate([image_plt, gt_mask, pr_mask]):
     axes[i*3+j].imshow(np.squeeze(im), cmap = 'gray')
     axes[i*3+j].set_xticks([])
     axes[i*3+j].set_yticks([])

plt.savefig('result.jpg', dpi=150)
plt.show()