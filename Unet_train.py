# -*- coding: utf-8 -*-
"""
@ Time:     2024/4/10 14:20 2024
@ Author:   CQshui
$ File:     Unet_train.py
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


# 数据集类
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


# 定义权重调整模块
class WeightAdjustmentModule(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        super(WeightAdjustmentModule, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(num_classes, (3, 3), padding='same')
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.global_pool(x)
        weights = self.fc(x)
        return weights


# 将权重调整模块嵌入到UNet模型
class AdaptiveUNet(tf.keras.Model):
    def __init__(self, unet_model, num_classes):
        super(AdaptiveUNet, self).__init__()
        self.unet = unet_model
        self.weight_module = WeightAdjustmentModule(num_classes)

    def call(self, inputs):
        predictions = self.unet(inputs)
        weights = self.weight_module(predictions)
        return predictions, weights


# 定义自适应损失
def adaptive_loss(predictions, targets, weights):
    # 计算Dice损失和Focal损失
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss()
    total_loss = dice_loss(predictions, targets) + focal_loss(predictions, targets)

    # 根据权重加权
    weighted_loss = tf.reduce_mean(total_loss * weights)
    return weighted_loss


# 数据增强函数
def get_augmentation():
    transforms = [
          A.HorizontalFlip(p=0.5),
          A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
          A.RandomBrightnessContrast(contrast_limit=0.3, brightness_limit=0.3, p=0.2),
          A.OneOf([
                A.ImageCompression(p=0.8),
                A.RandomGamma(p=0.8),
                A.Blur(p=0.8),
            ], p=1.0),
          A.OneOf([
                A.ImageCompression(p=0.8),
                A.RandomGamma(p=0.8),
                A.Blur(p=0.8),
            ], p=1.0),
          A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=cv2.BORDER_CONSTANT),
      ]

    return A.Compose(transforms)


def get_preprocessing(preprocessing_fn):
    _transform = [
            A.Lambda(image=preprocessing_fn),
        ]
    return A.Compose(_transform)


if __name__ == "__main__":
    train_images_dir = 'K:\pythonProject\Torch_test\datas/train\images'
    train_maps_dir = 'K:\pythonProject\Torch_test\datas/train\segmaps'
    val_images_dir = 'K:\pythonProject\Torch_test\datas/val/val_images'
    val_maps_dir = 'K:\pythonProject\Torch_test\datas/val/val_segmaps'
    test_images_dir = 'K:\pythonProject\Torch_test\datas/test/test_images'
    test_maps_dir = 'K:\pythonProject\Torch_test\datas/test/test_segmaps'


    dataset = Dataset(train_images_dir, train_maps_dir, augmentation = get_augmentation())

    sample_img, sample_map = dataset[2]

    fig, ax = plt.subplots(1,2, figsize = (15,10))
    ax[0].imshow(sample_img)
    ax[1].imshow(np.squeeze(sample_map), cmap = 'gray')
    plt.show()

    # 开始训练
    sm.set_framework('tf.keras')
    BACKBONE = 'efficientnetb3'
    ACTIVATION = 'sigmoid'
    BATCH_SIZE = 8
    NUM_CLASSES = 1
    LR = 1e-4
    EPOCHS = 20


    preprocess_input = sm.get_preprocessing(BACKBONE)
    model = sm.Unet(BACKBONE, classes=NUM_CLASSES, activation=ACTIVATION)

    optim = tf.keras.optimizers.Adam(LR)
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # 原模型
    # model.compile(optim, total_loss, metrics)
    # 自适应修改模型
    adaptive_unet = AdaptiveUNet(model, NUM_CLASSES)
    # 编译模型，使用自适应损失函数
    adaptive_unet.compile(optim, adaptive_loss, metrics)

    # Dataset for train images
    train_dataset = Dataset(
        train_images_dir,
        train_maps_dir,
        augmentation=get_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        val_images_dir,
        val_maps_dir,
        augmentation=get_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    # check shapes for errors
    assert train_dataloader[0][0].shape == (BATCH_SIZE, 512, 512, 3)
    assert train_dataloader[0][1].shape == (BATCH_SIZE, 512, 512, NUM_CLASSES)

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('temp/best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau()
    ]
    history = adaptive_unet.fit_generator(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
    )

    # 绘制损失函数图
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    plt.savefig('temp/loss.png', dpi=150)

    # 模型评价
    test_dataset = Dataset(
        test_images_dir,
        test_maps_dir,
        preprocessing=get_preprocessing(preprocess_input),
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    adaptive_unet.load_weights('temp/best_model.h5')
    scores = adaptive_unet.evaluate_generator(test_dataloader)

    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
        print("mean {}: {:.5}".format(metric.__name__, value))


    n = 5
    ids = np.random.choice(np.arange(len(test_dataset)), size=n)

    fig, axes = plt.subplots(n, 3, figsize = (16,25))

    cols = ['Image', "GT Box", "Pred"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    axes = axes.ravel()
    for i,id in enumerate(ids):
      image, gt_mask = test_dataset[id]
      image = np.expand_dims(image, axis=0)
      pr_mask = model.predict(image).round()

      # 由于用cv2读取，plt输出
      image_plt = cv2.cvtColor(np.squeeze(image), cv2.COLOR_RGB2GRAY)

      for j, im in enumerate([image, gt_mask, pr_mask]):
         axes[i*3+j].imshow(np.squeeze(im), cmap = 'gray')

    plt.savefig('result.jpg', dpi=150)
    plt.show()