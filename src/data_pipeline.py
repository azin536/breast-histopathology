import cv2
import omegaconf
import random

import numpy as np
import tensorflow as tf

from imgaug import augmenters as iaa
from tensorflow.keras.utils import Sequence
from typing import List


class DataGenerator(Sequence):
    def __init__(self, config: omegaconf.dictconfig.DictConfig, paths: List,
                labels: List, steps: int):
        self.config = config
        self.steps = steps
        self.x_path = paths
        self.batch_size = config.data_pipeline.batch_size
        self.target_size = config.data_pipeline.target_size
        self.labels = labels
        
    def __bool__(self):
        return True
    
    def __len__(self):
        return self.steps
    
    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx * self.batch_size: (1 + idx) * self.batch_size]
        batch_x = np.asarray([self.load_image(x_path) for x_path in batch_x_path])
        batch_x = self.transform_batch_images(batch_x)
        batch_y = self.labels[idx * self.batch_size: (1 + idx) * self.batch_size]
        batch_y = np.expand_dims(np.asarray(batch_y), axis=-1)
        return batch_x, batch_y
    
    def load_image(self, x_path):
        image = self.read_png(x_path)
        if image.shape != self.target_size:
            image_array = cv2.resize(image, self.target_size)
        else:
            image_array = image
        return image_array
        
    @staticmethod
    def read_png(x_path):
        image = tf.image.decode_png(tf.io.read_file(x_path))
        return np.asarray(image) / 255
    
    @staticmethod
    def transform_batch_images(batch_x):
        sometimes = lambda aug: iaa.Sometimes(1.0, aug)
        augmenter = iaa.Sequential(
                            [
                              iaa.Fliplr(0.5),
                              iaa.Flipud(0.5),
#                               iaa.geometric.Affine(rotate=(-20, 20), order=1, mode='constant', fit_output=False),
#                               sometimes(iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, shear=(-16, 16))),
#                               sometimes(iaa.Crop(px=(0, 25), keep_size=True, sample_independently=False)),
#                               sometimes(iaa.LinearContrast(0.5, 0.75))
                            ],
                            random_order=True,
                        )

        batch_x = augmenter.augment_images(batch_x)
        return batch_x
    
    def on_epoch_end(self):
        return random.shuffle(self.x_path)
