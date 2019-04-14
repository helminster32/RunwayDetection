import tensorflow as tf
import numpy as np
import os
import sys
import gflags

from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K

import logz
import nets
import utils
import data_utils
import log_utils
from common_flags import FLAGS


# Constants
TRAIN_PHASE = 1


def getModel(img_width, img_height, img_channels, output_dim, weights_path):

    model = nets.resnet50(img_width, img_height, img_channels, output_dim)

    if weights_path:
        try:
            model.load_weights(weights_path)
            print("Loaded model from {}".format(weights_path))
        except:
            print("Impossible to find weight path. Returning untrained model")

    return model


def trainModel(train_data_generator, val_data_generator, model, initial_epoch):

    # Configure training process
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  decay=1e-3,
                  metrics=['categorical_accuracy'])

    # Save model with the lowest validation loss
    weights_path = os.path.join(FLAGS.experiment_rootdir, 'weights_{epoch:03d}.h5')
    writeBestModel = ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                                     save_best_only=True, save_weights_only=True)

    # Save training and validation losses.
    logz.configure_output_dir(FLAGS.experiment_rootdir)
    saveModelAndLoss = log_utils.MyCallback(filepath=FLAGS.experiment_rootdir)

    # Train model
    steps_per_epoch = int(np.ceil(train_data_generator.samples / FLAGS.batch_size))
    validation_steps = int(np.ceil(val_data_generator.samples / FLAGS.batch_size))-1

    model.fit_generator(train_data_generator,
                        epochs=FLAGS.epochs, steps_per_epoch = steps_per_epoch,
                        callbacks=[writeBestModel, saveModelAndLoss],
                        validation_data=val_data_generator,
                        validation_steps = validation_steps,
                        initial_epoch=initial_epoch)


def _main():
    
    # Set random seed
    if FLAGS.random_seed:
        seed = np.random.randint(0,2*31-1)
    else:
        seed = 5
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Set training phase
    K.set_learning_phase(TRAIN_PHASE)

    # Create the experiment rootdir if not already there
    if not os.path.exists(FLAGS.experiment_rootdir):
        os.makedirs(FLAGS.experiment_rootdir)

    # Input image dimensions
    img_width, img_height = FLAGS.img_width, FLAGS.img_height
    
    # Image mode (RGB or grayscale)
    if FLAGS.img_mode=='rgb':
        img_channels = 3
    elif FLAGS.img_mode == 'grayscale':
        img_channels = 1
    else:
        raise IOError("Unidentified image mode: use 'grayscale' or 'rgb'")

    # Output dimension (4 coordinates 8 values)
    num_coordinates = 8
       

    # Generate training data with real-time augmentation
    train_datagen = data_utils.DataGenerator(rescale = 1./255)
    
    # Iterator object containing training data to be generated batch by batch
    train_generator = train_datagen.flow_from_file(FLAGS.train_dir,
                                                        num_coordinates,
                                                        shuffle = True,
                                                        img_mode = FLAGS.img_mode,
                                                        target_size=(img_height, img_width),
                                                        batch_size = FLAGS.batch_size)

    # Generate validation data with real-time augmentation
    val_datagen = data_utils.DataGenerator(rescale = 1./255)
    
    # Iterator object containing validation data to be generated batch by batch
    val_generator = val_datagen.flow_from_file(FLAGS.val_dir,
                                                    num_coordinates,
                                                    shuffle = False,
                                                    img_mode = FLAGS.img_mode,
                                                    target_size=(img_height, img_width),
                                                    batch_size = FLAGS.batch_size)

    # Weights to restore
    weights_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.weights_fname)
    
    # Epoch from which training starts
    initial_epoch = 0
    if not FLAGS.restore_model:
        # In this case weights are initialized randomly
        weights_path = None
    else:
        # In this case weigths are initialized as specified in pre-trained model
        initial_epoch = FLAGS.initial_epoch

    # Define model
    model = getModel(img_width, img_height, img_channels,
                        num_coordinates, weights_path)

    # Save the architecture of the network as png
    plot_arch_path = os.path.join(FLAGS.experiment_rootdir, 'architecture.png')
    plot_model(model, to_file=plot_arch_path)

    # Serialize model into json
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    utils.modelToJson(model, json_model_path)

    # Train model
    trainModel(train_generator, val_generator, model, initial_epoch)
    
    # Plot training and validation losses
    utils.plot_loss(FLAGS.experiment_rootdir)


def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))

      sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
