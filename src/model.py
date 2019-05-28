import tensorflow as tf
import numpy as np
import os
from numpy import genfromtxt
import pickle
from tensorflow.python.lib.io import file_io
import subprocess
import sys
import pandas as pd



ENV = 'gcp'
BUCKET = None  # set from task.py
PATTERN = '*' # gets all files

#Hyperparameters
TRAIN_STEPS = 10000
EVAL_STEPS = None
BATCH_SIZE = 256

CSV_COLUMNS = ['signal_observation']
INPUT_COLUMNS = [
    tf.feature_column.numeric_column('signal_observation')
]
DEFAULTS = [[0.0]]
def read_dataset(prefix, mode, batch_size = 512):
    def load_and_preprocess_signal(path, label):
        #print(path)
        return np.reshape(genfromtxt(file_io.FileIO(path.decode(), 'r'),dtype='float32', delimiter=','), (150, 1000)), label

    print('Reading {} data.'.format(prefix))

    labels_ds = None
    file_list = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        label_file_path = 'gs://lanl-earthquake-gpu-large/{}_labels.pkl'.format(prefix)
        with file_io.FileIO(label_file_path, 'rb') as f:
            labels = pickle.load(f)

        file_list = ['{}/'.format(prefix) + i for i in labels.keys()]
        labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(list(labels.values()), tf.float32))
    else:
        file_list = ['{}/'.format(prefix) + i for i in os.listdir('{}/'.format(prefix))]
        with file_io.FileIO('./data/15000_processed_data/file_names.pkl', 'wb') as f:
            pickle.dump(file_list, f)
        labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast([0.0 for i in range(len(file_list))], tf.float32))

    path_ds = tf.data.Dataset.from_tensor_slices(file_list)
    dataset = tf.data.Dataset.zip((path_ds, labels_ds))
    dataset = dataset.map(lambda filename, label: tuple(tf.py_func(load_and_preprocess_signal, [filename, label], [tf.float32, label.dtype])))

    if mode == tf.estimator.ModeKeys.TRAIN:
        num_epochs = None # indefinitely
        dataset = dataset.shuffle(buffer_size = 10 * batch_size)
    else:
        num_epochs = 1 # end-of-input after this

    dataset = dataset.repeat(num_epochs).batch(batch_size)

    return dataset

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    features = tf.reshape(features, [-1, 150000, 1])
    labels = tf.reshape(labels, [-1, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv1d(
      inputs=features,
      filters=4,
      kernel_size=500,
      padding="same",
      activation=tf.nn.leaky_relu)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv1d(
      inputs=pool1,
      filters=8,
      kernel_size=250,
      padding="same",
      activation=tf.nn.leaky_relu)
    pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)

    conv3 = tf.layers.conv1d(
      inputs=pool2,
      filters=16,
      kernel_size=125,
      padding="same",
      activation=tf.nn.leaky_relu)
    pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2)

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMBlockCell(16), tf.contrib.rnn.LSTMBlockCell(8), tf.contrib.rnn.LSTMBlockCell(4)])

    init_state = stacked_lstm.zero_state(tf.shape(features)[0], tf.float32)
    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, pool3,
                                   initial_state=init_state,
                                   dtype=tf.float32)


    #ACCOUNT FOR BATCH_SIZE HERE
    dense_1 = tf.layers.dense(inputs=outputs[:, -1, :], units=10)
    predictions = tf.layers.dense(inputs=dense_1, units=1)


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels, predictions)

    accuracy = tf.metrics.mean_absolute_error(labels=labels, predictions=predictions)
    tf.summary.scalar('MAE', accuracy[1])
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss)


def train_and_evaluate(output_dir):
    eval_filename = 'eval.tar.gz'
    train_filename = 'train.tar.gz'

    subprocess.check_call(['gsutil', 'cp', os.path.join('gs://lanl-earthquake-gpu-large', eval_filename), eval_filename], stderr=sys.stdout)
    subprocess.check_call(['gsutil', 'cp', os.path.join('gs://lanl-earthquake-gpu-large', train_filename), train_filename], stderr=sys.stdout)

    subprocess.call(['tar', '-xf', eval_filename])

    subprocess.call(['tar', '-xf', train_filename])

    EVAL_INTERVAL = 3600

    run_config = tf.estimator.RunConfig(save_summary_steps=10, save_checkpoints_secs = EVAL_INTERVAL,
                                      keep_checkpoint_max = 3)
    signal_regressor = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=output_dir, config = run_config)

    hook = tf.contrib.estimator.stop_if_no_decrease_hook(signal_regressor, 'loss', 300)

    train_spec = tf.estimator.TrainSpec(
                       input_fn = lambda: read_dataset("train", mode=tf.estimator.ModeKeys.TRAIN, batch_size=BATCH_SIZE),
                       max_steps = TRAIN_STEPS, hooks=[hook])

    eval_spec = tf.estimator.EvalSpec(
                       input_fn = lambda: read_dataset("eval", mode=tf.estimator.ModeKeys.EVAL, batch_size=BATCH_SIZE))

    tf.estimator.train_and_evaluate(signal_regressor, train_spec, eval_spec)

def predict(output_dir):
    signal_regressor = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=output_dir)

    predictions = list()
    for i in signal_regressor.predict(lambda: read_dataset("./data/15000_processed_data/test", mode=tf.estimator.ModeKeys.PREDICT, batch_size=BATCH_SIZE)):
        predictions.append(i)
        print(i)



    with file_io.FileIO('predictions.pkl', 'wb') as f:
        pickle.dump(predictions, f)

    with file_io.FileIO('./data/15000_processed_data/file_names.pkl', 'rb') as f:
        file_list = pickle.load(f)

    df = pd.DataFrame(list(zip([i.split('/')[4][:-4] for i in file_list], [i[0] if i[0] < 0.0 else 0.0 for i in predictions])), columns=['seg_id', 'time_to_failure'])
    df['time_to_failure'] = df['time_to_failure'].apply(lambda x: 0.0 if x < 0.0 else x)
    df.to_csv('./data/15000_processed_data/submission_standard.csv', index=False)
