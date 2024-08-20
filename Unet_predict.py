# -*- coding: utf-8 -*-
"""
@ Time:     2024/4/10 18:06 2024
@ Author:   CQshui
$ File:     Unet_predict.py
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
    def __init__(
                 self,
                 images_dir,
                 preprocessing=None
                 ):

        self.img_ids = os.listdir(images_dir)

        self.img_fps = [os.path.join(images_dir, id) for id in self.img_ids]

        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_fps[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shape_store.append(image.shape[:2])
        image = cv2.resize(image, (512, 512))


        if self.preprocessing:
            processed_sample = self.preprocessing(image=image, mask=image)
            image = processed_sample['image']

        print(image.dtype())

        return image, image.astype(np.float32)

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



if __name__ == "__main__":
    predict_images_dir = r'F:\Data\20240713\fangjieshi\4\0.00012\phase'
    output_dir = r'F:\Data\20240713\fangjieshi\4\0.00012\binary'
    shape_store = []

    sm.set_framework('tf.keras')
    BACKBONE = 'efficientnetb3'
    ACTIVATION = 'sigmoid'
    NUM_CLASSES = 1


    preprocess_input = sm.get_preprocessing(BACKBONE)
    model = sm.Unet(BACKBONE, classes=NUM_CLASSES, activation=ACTIVATION)

    test_dataset = Dataset(
        predict_images_dir,
        preprocessing=get_preprocessing(preprocess_input)
    )


    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.load_weights('best_model.h5')

    n = 5
    ids = [x for x in range(len(test_dataset))]
    # ids = np.random.choice(np.arange(len(test_dataset)), size=n)

    fig, axes = plt.subplots(n, 2, figsize = (16,25))

    cols = ['Image', "Pred"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    axes = axes.ravel()

    for i,id in enumerate(ids):
        image = test_dataset[id][0]

        image = np.expand_dims(image, axis=0)
        image_id = test_dataset.img_ids[id]
        pr_mask = model.predict(image).round()
        pr_mask = np.squeeze(pr_mask)*255
        pr_mask = cv2.resize(pr_mask, (shape_store[id][1], shape_store[id][0]))
        cv2.imwrite(output_dir + r'\{}'.format(image_id), pr_mask)






    # for j, im in enumerate([image, pr_mask]):
    #     axes[i*2+j].imshow(np.squeeze(im), cmap = 'gray')
    #     print(np.squeeze(im).shape)
# plt.savefig('result.jpg', dpi=150)
# plt.show()

# predict_img_dir = 'C:/Users/d1009/Desktop/archive0/images/62223ab32f.png'
# image = cv2.imread(predict_img_dir)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = cv2.resize(image, (512, 512))
# image = np.expand_dims(image, axis=0)
# image = image.astype(np.float32) / 255.0
#
# pr_mask = model.predict(image).round()
#
# cv2.imwrite('image.jpg', np.squeeze(pr_mask)*255)


