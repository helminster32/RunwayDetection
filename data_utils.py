import os
import numpy as np
import cv2
import pandas as pd
import random

import keras
from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator(ImageDataGenerator):

    def flow_from_file(self, file, num_coordinates, target_size=(224, 224),
                            img_mode='grayscale', batch_size=32, shuffle=True,
                            seed=None):

        return FileIterator(
            file, num_coordinates, self, target_size=target_size, img_mode=img_mode,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            )

class FileIterator(Iterator):

        def __init__(self, file, num_coordinates, image_data_generator,
                 target_size=(224, 224), img_mode='grayscale',
                 batch_size=32, shuffle=True, seed=None):

            self.file = file
            self.image_data_generator = image_data_generator
            self.target_size = tuple(target_size)

            # Initialize image mode
            if img_mode not in {'rgb', 'grayscale'}:
                raise ValueError('Invalid color mode:', img_mode,
                                 '; expected "rgb" or "grayscale".')
            self.img_mode = img_mode
            if self.img_mode == 'rgb':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = self.target_size + (1,)

            # Initialize number of classes
            self.num_coordinates = num_coordinates

            # Allowed image formats
            self.formats = {'png', 'jpg'}

            # Filenames of all samples/images in dataset.
            self.filenames = []
            # Labels (ground truth) of all samples/images in dataset
            self.ground_truth = []

            # Open csv
            self.data = pd.read_csv(self.file)

            # Conversion of list into array
            self.ground_truth = np.array(self.ground_truth, dtype=K.floatx())

            super(FileIterator, self).__init__(
                    batch_size, shuffle, seed)

        def next(self):

            with self.lock:
                index_array, current_index, current_batch_size = next(
                    self.index_generator)

            # Initialize batch of images
            batch_x = np.zeros((current_batch_size,) + self.image_shape,
                               dtype=K.floatx())
            # Initialize batch of ground truth
            batch_y = np.zeros((current_batch_size, self.num_coordinates,),
                               dtype=K.floatx())

            grayscale = self.img_mode == 'grayscale'

            # Build batch of image data
            # Extract random indexes
            rows = random.randint(0, self.data.shape[0] - 1, current_batch_size)

            for i in rows:
                # Leer de mi csv el path y el nombre de la imagen
                x = load_img(self.data.iloc(i, 0),
                             grayscale=grayscale,
                             target_size=self.target_size)
                # Data augmentation
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i] = x
                # Leer de mi csv las coordenadas
                batch_y[i] = self.data.iloc(i, 1)
            '''
            # Build batch of labels
            batch_y = np.array(self.ground_truth[index_array], dtype=K.floatx())
            batch_y = keras.utils.to_categorical(batch_y, num_classes=self.num_classes)
            '''

            return batch_x, batch_y

def load_img(path, grayscale=False, target_size=None):

    # Read input image
    img = cv2.imread(path)

    if grayscale:
       if len(img.shape) != 2:
          img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if target_size:
        if (img.shape[0], img.shape[1]) != target_size:
            img = cv2.resize(img, (target_size[1], target_size[0]))

    if grayscale:
        img = img.reshape((img.shape[0], img.shape[1], 1))

    return np.asarray(img, dtype=np.float32)


