import os
import numpy as np
import cv2
import re

import keras
from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator(ImageDataGenerator):
    """
    Generate minibatches of images and labels with real-time augmentation.

    The only function that changes w.r.t. parent class is the flow that
    generates data. This function needed in fact adaptation for different
    directory structure and labels. All the remaining functions remain
    unchanged.
    """

    def flow_from_directory(self, directory, coordinates, target_size=(224, 224),
                            img_mode='grayscale', batch_size=32, shuffle=True,
                            seed=None):

        return DirectoryIterator(
            directory, coordinates, self, target_size=target_size, img_mode=img_mode,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            )
    """
    Cosas a cambiar desde aquí num_classes tendría que ser points o algo así y el follow links
        sobra también
    """

class DirectoryIterator(Iterator):
    """
        Class for managing data loading of images and labels

        # Arguments
           file: Path to the file to read data from.
           coordinates: Output dimension.
           image_data_generator: Image Generator.
           target_size: tuple of integers, dimensions to resize input images to.
           img_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
           batch_size: The desired batch size
           shuffle: Whether to shuffle data or not
           seed : numpy seed to shuffle data

        # TODO: Add functionality to save images to have a look at the augmentation
        """

        def __init__(self, file, coordinates, image_data_generator,
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
            self.num_classes = num_classes

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


    def _decode_experiment_dir(self, image_dir_path):
        """
        Extract valid filenames in every class.

        # Arguments
            image_dir_path: path to class folder to be decoded
        """
        for root, _, files in self._recursive_list(image_dir_path):
            self.samples_per_class.append(len(files))
            for frame_number, fname in enumerate(files):
                is_valid = False
                for extension in self.formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    absolute_path = os.path.join(root, fname)
                    self.filenames.append(os.path.relpath(absolute_path,
                                                          self.directory))
                    self.samples += 1


    def next(self):
        """
        Public function to fetch next batch.

        Image transformation is not under thread lock, so it can be done in
        parallel

        # Returns
            The next batch of images and categorical labels.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)

        # Initialize batch of images
        batch_x = np.zeros((current_batch_size,) + self.image_shape,
                           dtype=K.floatx())
        # Initialize batch of ground truth
        batch_y = np.zeros((current_batch_size, self.num_classes,),
                           dtype=K.floatx())

        grayscale = self.img_mode == 'grayscale'

        # Build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = load_img(os.path.join(self.directory, fname),
                         grayscale=grayscale,
                         target_size=self.target_size)
            # Data augmentation
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        # Build batch of labels
        batch_y = np.array(self.ground_truth[index_array], dtype=K.floatx())
        batch_y = keras.utils.to_categorical(batch_y, num_classes=self.num_classes)

        return batch_x, batch_y


def load_img(path, grayscale=False, target_size=None):
    """
    Load an image.

    # Arguments
        path: Path to image file.
        grayscale: Boolean, wether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.

    # Returns
        Image as numpy array.
    """

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


