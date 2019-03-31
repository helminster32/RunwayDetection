import os
import numpy as np
import cv2
import re

import keras
from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator(ImageDataGenerator):

    def flow_from_file(self, file, num_coordinates, target_size=(224, 224),
                            img_mode='grayscale', batch_size=32, shuffle=True,
                            seed=None):

        return DirectoryIterator(
            file, num_coordinates, self, target_size=target_size, img_mode=img_mode,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            )

class DirectoryIterator(Iterator):

        def __init__(self, file, num_coordinates, image_data_generator,
                 target_size=(224, 224), img_mode='grayscale',
                 batch_size=32, shuffle=True, seed=None):
            self.file = os.path.realpath(file)
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

            # Number of samples in dataset
            self.samples = 0

            # Filenames of all samples/images in dataset.
            self.filenames = []
            # Labels (ground truth) of all samples/images in dataset
            self.ground_truth = []

            filepath = file

            with open(filepath) as fp:
                for linea in enumerate(fp):
                    a = str(linea)
                    x1 = int(re.search(r'(?<=x"":\[)[0-9]*', a).group())
                    x2 = int(re.search(r'(?<=x"":\[[0-9]{3},)[0-9]{3}', a).group())
                    x3 = int(re.search(r'(?<=x"":\[[0-9]{3},[0-9]{3},)[0-9]{3}', a).group())
                    x4 = int(re.search(r'(?<=x"":\[[0-9]{3},[0-9]{3},[0-9]{3},)[0-9]{3}', a).group())

                    y1 = int(re.search(r'(?<=y"":\[)[0-9]*', a).group())
                    y2 = int(re.search(r'(?<=y"":\[[0-9]{3},)[0-9]{3}', a).group())
                    y3 = int(re.search(r'(?<=y"":\[[0-9]{3},[0-9]{3},)[0-9]{3}', a).group())
                    y4 = int(re.search(r'(?<=y"":\[[0-9]{3},[0-9]{3},[0-9]{3},)[0-9]{3}', a).group())

                    self.ground_truth.append([x1, x2, x3, x4, y1, y2, y3, y4])

                    imagename = ".\\DataBase\\" + re.search(r'[0-9]*.jpg', a).group()
                    CurrentImage = cv2.imread(imagename)

                    self.filenames.append(np.asarray(CurrentImage))

            # Conversion of list into array
            self.ground_truth = np.array(self.ground_truth, dtype=K.floatx())

        super(DirectoryIterator, self).__init__(self.samples,
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
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = load_img(os.path.join(self.file, fname),
                         grayscale=grayscale,
                         target_size=self.target_size)
            # Data augmentation
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        # Build batch of labels
        batch_y = np.array(self.ground_truth[index_array], dtype=K.floatx())

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


