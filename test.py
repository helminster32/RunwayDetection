import gflags
import numpy as np
import os
import sys
from sklearn import metrics

from keras import backend as K

import utils
import data_utils
from common_flags import FLAGS 

# Constants
TEST_PHASE = 0


            
def compute_highest_classification_errors(pred_probs, real_labels, n_errors=20):

    assert np.all(pred_probs.shape == real_labels.shape)
    dist = abs(pred_probs - 1)
    highest_errors = dist.argsort()[-n_errors:][::-1]
    return highest_errors


def evaluate_classification(pred_probs, pred_labels, real_labels):

    # Compute average accuracy
    ave_accuracy = metrics.accuracy_score(real_labels, pred_labels)
    print('Average accuracy = ', ave_accuracy)
    
    # Compute highest errors
    highest_errors = compute_highest_classification_errors(pred_probs, real_labels,
            n_errors=20)
    
    # Return accuracy and highest errors in a dictionary
    dictionary = {"ave_accuracy": ave_accuracy.tolist(),
                  "highest_errors": highest_errors.tolist()}
    return dictionary


def _main():

    # Set testing mode (dropout/batchnormalization)
    K.set_learning_phase(TEST_PHASE)
    
    # Output dimension (8 coordinates)
    num_coordinates = 8

    # Generate testing data
    test_datagen = data_utils.DataGenerator(rescale=1./255)
    
    # Iterator object containing testing data to be generated batch by batch
    test_generator = test_datagen.flow_from_directory(FLAGS.test_dir,
                                                      num_classes,
                                                      shuffle=False,
                                                      img_mode=FLAGS.img_mode,
                                                      target_size=(FLAGS.img_height, FLAGS.img_width),
                                                      batch_size = FLAGS.batch_size)

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    model = utils.jsonToModel(json_model_path)

    # Load weights
    weights_load_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.weights_fname)
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except:
        print("Impossible to find weight path. Returning untrained model")


    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Get predictions and ground truth
    n_samples = test_generator.samples
    nb_batches = int(np.ceil(n_samples / FLAGS.batch_size))
    probs_per_class, ground_truth = utils.compute_predictions_and_gt(
            model, test_generator, nb_batches, verbose = 1)
    
    # Predicted probabilities
    pred_probs = np.max(probs_per_class, axis=-1)
    # Prediced labels
    pred_labels = np.argmax(probs_per_class, axis=-1)
    # Real labels (ground truth)
    real_labels = np.argmax(ground_truth, axis=-1)
          
                  
    # Evaluate predictions: Average accuracy and highest errors
    print("-----------------------------------------------")
    print("Evalutaion:")
    evaluation = evaluate_classification(pred_probs, pred_labels, real_labels)
    print("-----------------------------------------------")
    
    # Save evaluation
    utils.write_to_file(evaluation, os.path.join(FLAGS.experiment_rootdir, 'test_results.json'))

    # Save predicted and real steerings as a dictionary
    labels_dict = {'pred_labels': pred_labels.tolist(),
                  'real_labels': real_labels.tolist()}
    utils.write_to_file(labels_dict, os.path.join(FLAGS.experiment_rootdir,
                                               'predicted_and_real_labels.json'))
                                               
    # Visualize confusion matrix                                           
    utils.plot_confusion_matrix(FLAGS.experiment_rootdir, real_labels, pred_labels,
                                CLASSES, normalize=True)

                                               
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