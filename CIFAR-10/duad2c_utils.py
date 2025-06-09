
import numpy as np
import random
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K



def load(paths, verbose=-1):
    '''expects images/sequences for each class in seperate dir, 
    e.g all digits in 0 class in the directory named 0 '''
    data = list()
    labels = list()
    # loop over the input images/sequences
    for (i, imgpath) in enumerate(paths):
        # load the image/sequence and extract the class labels
        im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray).flatten()
        label = imgpath.split(os.path.sep)[-2]
        # scale the image/sequence to [0, 1] and add to list
        data.append(image/255)
        labels.append(label)
        # show an update every `verbose` images/sequences
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
    # return a tuple of the data and labels
    return data, labels


def create_subsets(image_list, label_list, num_subsets=3, initial='subset'):
    ''' return: a dictionary with keys subset names and value as
                data shards - tuple of sequence and label lists.
        args:
            image_list: a list of numpy arrays of inputs
            label_list:a list of binarized labels for each sequence
            num_subsets: number of training subsets
            initials: the subset name prefix, e.g, subset_1

    '''

    #create a list of subset no.
    subset_no = ['{}_{}'.format(initial, i+1) for i in range(num_subsets)]

    #randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    #shard data and place at each subset
    size = len(data)//num_subsets
    shards = [data[i:i + size] for i in range(0, size*num_subsets, size)]

    #number of subsets must equal number of shards
    assert(len(shards) == len(subset_no))

    return {subset_no[i] : shards[i] for i in range(len(subset_no))}



def batch_data(data_shard, bs=32):
    '''Takes in a subsets' data shard and creates a tfds object off it
    args:
        shard: a data, label constituting a subsets' data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)
    

#def weight_scalling_factor(subset_trn_data, subset_no, subset_model=None, X_val=None, y_val=None, alpha=0.7):
#    '''Dynamically scales weights using validation accuracy and uncertainty.'''
#    if subset_model is None or X_val is None or y_val is None:
#        # Fallback to data-size-based scaling if validation args are missing
#        bs = list(subset_trn_data[subset_no])[0][0].shape[0]
#        central_data_count = sum([tf.data.experimental.cardinality(subset_trn_data[s]).numpy()*bs for s in subset_trn_data])
#        subset_data_count = tf.data.experimental.cardinality(subset_trn_data[subset_no]).numpy()*bs
#        return subset_data_count / central_data_count
#    else:
#        # 1. Calculate validation accuracy
#        _, acc = subset_model.evaluate(X_val, y_val, verbose=0)
#        
#        # 2. Compute prediction entropy (uncertainty)
#        logits = subset_model.predict(X_val, verbose=0)
#        probas = tf.nn.softmax(logits)
#        entropy = -tf.reduce_sum(probas * tf.math.log(probas + 1e-8), axis=1)
#        avg_entropy = tf.reduce_mean(entropy).numpy()
        
#        # 3. Combine metrics
#        uncertainty_score = 1 / (avg_entropy + 1e-8)
#        composite_scale = (alpha * acc) + ((1 - alpha) * uncertainty_score)
#        return composite_scale
        
#def weight_scalling_factor(subset_trn_data, subset_no, subset_model=None, 
#                          X_val=None, y_val=None, alpha=0.5, epsilon=1e-8):
#    '''Dynamically scales weights using normalized accuracy and uncertainty.'''
#    if subset_model is None or X_val is None or y_val is None:
#        # Fallback to data-size-based scaling
#        bs = list(subset_trn_data[subset_no])[0][0].shape[0]
#        central_data_count = sum([tf.data.experimental.cardinality(subset_trn_data[s]).numpy()*bs 
#                            for s in subset_trn_data])
#        subset_data_count = tf.data.experimental.cardinality(subset_trn_data[subset_no]).numpy()*bs
#        return subset_data_count / central_data_count
#    
#    else:
#        # 1. Calculate validation accuracy
#        try:
#            _, acc = subset_model.evaluate(X_val, y_val, verbose=0, batch_size=256)
#        except:
#            acc = 0.0  # Fallback if validation fails
#
#        # 2. Compute stabilized entropy
#        logits = subset_model.predict(X_val, verbose=0, batch_size=256)
#        
#        # Clip logits to prevent extreme softmax outputs
#        logits = tf.clip_by_value(logits, -20.0, 20.0)
#        probas = tf.nn.softmax(logits)
#        
#        # Clip probabilities and avoid log(0)
#        probas = tf.clip_by_value(probas, epsilon, 1.0 - epsilon)
#        entropy = -tf.reduce_sum(probas * tf.math.log(probas), axis=1)
#        
#        # 3. Stabilize entropy values
#        avg_entropy = tf.reduce_mean(entropy).numpy()
#        avg_entropy = np.clip(avg_entropy, epsilon, 100.0)  # Cap max entropy
#        
#        # 4. Compute uncertainty score and normalize to [0, 1]
#        uncertainty_score = 1 / (avg_entropy + epsilon)
#        print("uncertainty: ", uncertainty_score)
#        uncertainty_score = np.clip(uncertainty_score, 0.001, 20.001)
#        normalized_uncertainty = (uncertainty_score - 0.001) / 20.0  # Min-Max scaling
#        
#        # 5. Combine metrics with balanced contributions
#        composite_scale = (alpha * acc) + ((1 - alpha) * normalized_uncertainty)
#        
#        # Final failsafe
#        if np.isnan(composite_scale) or np.isinf(composite_scale):
#            composite_scale = 0.001  # Minimum contribution
#        
#        return composite_scale
        
def weight_scalling_factor(subset_trn_data, subset_no, subset_model=None,
                          X_val=None, y_val=None, alpha=0.5, epsilon=1e-8):
    '''Dynamically scales weights using normalized accuracy and uncertainty (GPU-friendly).'''
    if subset_model is None or X_val is None or y_val is None:
        # Fallback to data-size-based scaling (remains CPU-based if input is CPU data)
        bs = list(subset_trn_data[subset_no])[0][0].shape[0]
        central_data_count = sum([tf.data.experimental.cardinality(subset_trn_data[s]).numpy()*bs
                            for s in subset_trn_data])
        subset_data_count = tf.data.experimental.cardinality(subset_trn_data[subset_no]).numpy()*bs
        return subset_data_count / central_data_count

    else:
        # 1. Calculate validation accuracy
        try:
            _, acc = subset_model.evaluate(X_val, y_val, verbose=0, batch_size=256)
        except:
            acc = tf.constant(0.0, dtype=tf.float32)  # Use TensorFlow constant

        # 2. Compute stabilized entropy (Keep on GPU)
        logits = subset_model.predict(X_val, verbose=0, batch_size=256)
        logits = tf.clip_by_value(logits, -20.0, 20.0)
        probas = tf.nn.softmax(logits)
        probas = tf.clip_by_value(probas, epsilon, 1.0 - epsilon)
        entropy = -tf.reduce_sum(probas * tf.math.log(probas), axis=1)

        # 3. Stabilize entropy values (Keep on GPU)
        avg_entropy = tf.reduce_mean(entropy)
        avg_entropy = tf.clip_by_value(avg_entropy, epsilon, 100.0)

        # 4. Compute uncertainty score and normalize to [0, 1] (Keep on GPU)
        uncertainty_score = 1 / (avg_entropy + epsilon)
        print("uncertainty: ", uncertainty_score.numpy()) # Keep print for monitoring
        uncertainty_score = tf.clip_by_value(uncertainty_score, 0.001, 10.001)
        normalized_uncertainty = (uncertainty_score - 0.001) / 10.0

        # 5. Combine metrics with balanced contributions (Keep on GPU)
        composite_scale = (alpha * acc) + ((1 - alpha) * normalized_uncertainty)

        # Final failsafe (Keep on GPU)
        if tf.math.is_nan(composite_scale) or tf.math.is_inf(composite_scale):
            composite_scale = tf.constant(0.001, dtype=tf.float32)

        return composite_scale.numpy() # Return as NumPy for further CPU-based logic if needed

def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad


def test_model(X_test, Y_test, model, comm_round, return_entropy=False):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    
    if return_entropy:
        probas = tf.nn.softmax(logits)
        entropy = -tf.reduce_sum(probas * tf.math.log(probas + 1e-8), axis=1)
        avg_entropy = tf.reduce_mean(entropy).numpy()
        return acc, loss, avg_entropy
    else:
        print(f'comm_round: {comm_round} | global_acc: {acc:.3%} | global_loss: {loss}')
        return acc, loss

def test_train_model(X_test, Y_test,  model, comm_round, return_entropy=False):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    
    if return_entropy:
        probas = tf.nn.softmax(logits)
        entropy = -tf.reduce_sum(probas * tf.math.log(probas + 1e-8), axis=1)
        avg_entropy = tf.reduce_mean(entropy).numpy()
        return acc, loss, avg_entropy
    else:
        print('comm_round: {} | global_training_acc: {:.3%} | global_training_loss: {}'.format(comm_round, acc, loss))
        return acc, loss

def normalize_scaling_factors(scaling_factors):
    total_scale = sum(scaling_factors)
    return [s / total_scale for s in scaling_factors]