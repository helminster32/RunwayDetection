import numpy as np
import json
import os
import matplotlib.pyplot as plt
import itertools

from keras.utils.generic_utils import Progbar
from keras.models import model_from_json

from sklearn.metrics import confusion_matrix


def compute_predictions_and_gt(model, generator, steps,
                                     max_q_size=10,
                                     pickle_safe=False, verbose=0):
    """
    Generate predictions and associated ground truth for the input samples
    from a data generator. The generator should return the same kind of data as
    accepted by `predict_on_batch`.
    
    Function adapted from keras `predict_generator`.

    # Arguments
        model: Model instance containing the trained model.
        generator: Generator yielding batches of input samples.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
        max_q_size: Maximum size for the generator queue.
        pickle_safe: If `True`, use process based threading.
            Note that because
            this implementation relies on multiprocessing,
            you should not pass
            non picklable arguments to the generator
            as they can't be passed
            easily to children processes.
        verbose: verbosity mode, 0 or 1.

    # Returns
        Numpy array(s) of predictions and associated ground truth.

    # Raises
        ValueError: In case the generator yields
            data in an invalid format.
    """
    steps_done = 0
    all_outs = []
    all_steerings = []

    if verbose == 1:
        progbar = Progbar(target=steps)

    while steps_done < steps:
        generator_output = next(generator)

        if isinstance(generator_output, tuple):
            if len(generator_output) == 2:
                x, gt_steer = generator_output
            elif len(generator_output) == 3:
                x, gt_steer, _ = generator_output
            else:
                raise ValueError('output of generator should be '
                                 'a tuple `(x, y, sample_weight)` '
                                 'or `(x, y)`. Found: ' +
                                 str(generator_output))
        else:
            raise ValueError('Output not valid for current evaluation')

        outs = model.predict_on_batch(x)
        
        if not isinstance(outs, list):
            outs = [outs]
        if not isinstance(gt_steer, list):
            gt_steer = [gt_steer]

        if not all_outs:
            for out in outs:
            # Len of this list is related to the number of
            # outputs per model(1 in our case)
                all_outs.append([])

        if not all_steerings:
            # Len of list related to the number of gt_steerings
            # per model (1 in our case )
            for steer in gt_steer:
                all_steerings.append([])


        for i, out in enumerate(outs):
            all_outs[i].append(out)

        for i, steer in enumerate(gt_steer):
            all_steerings[i].append(steer)

        steps_done += 1
        if verbose == 1:
            progbar.update(steps_done)

    if steps_done == 1:
        return [out for out in all_outs], [steer for steer in all_steerings]
    else:
        return np.squeeze(np.array([np.concatenate(out) for out in all_outs])), \
                np.squeeze(np.array([np.concatenate(steer) for steer in all_steerings]))

    

def modelToJson(model, json_model_path):
    """
    Serialize model into json.
    """
    model_json = model.to_json()

    with open(json_model_path,"w") as f:
        f.write(model_json)



def jsonToModel(json_model_path):
    """
    Serialize json into model.
    """
    with open(json_model_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    return model



def write_to_file(dictionary, fname):
    """
    Writes everything is in a dictionary in json model.
    """
    with open(fname, "w") as f:
        json.dump(dictionary,f)
        print("Written file {}".format(fname))
        
        
def plot_loss(path_to_log):
    """
    Read log file and plot losses.
    
    # Arguments
        path_to_log: Path to log file.
    """
    # Read log file
    log_file = os.path.join(path_to_log, "log.txt")
    try:
        log = np.genfromtxt(log_file, delimiter='\t',dtype=None, names=True)
    except:
        raise IOError("Log file not found")

    train_loss = log['train_loss']
    val_loss = log['val_loss']
    timesteps = list(range(train_loss.shape[0]))
    
    # Plot losses
    plt.plot(timesteps, train_loss, 'r--', timesteps, val_loss, 'b--')
    plt.legend(["Training loss", "Validation loss"])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig(os.path.join(path_to_log, "log.png"))
    #plt.show()
    
    
def plot_confusion_matrix(path_to_results, real_labels, pred_labels, classes,
                          normalize=True):
    """
    Plot and save confusion matrix computed from predicted and real labels.
    
    # Arguments
        path_to_results: Location where saving confusion matrix.
        real_labels: List of real labels.
        pred_prob: List of predicted probabilities.
        normalize: Boolean, whether to apply normalization.
    """
    
    # Generate confusion matrix
    cm = confusion_matrix(real_labels, pred_labels)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=45)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, float('%.3f'%(cm[i, j])),
                 horizontalalignment="center")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(path_to_results, "confusion.png"))
    #plt.show()
