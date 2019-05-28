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

TRAIN_EXAMPLES = 6712
JOB_DIR='gs://lanl-earthquake-gpu-large'

def read_dataset(prefix, num_epochs, mode, batch_size = 512):
    def load_and_preprocess_signal(path, label):
        #print(path)
        features = np.reshape(genfromtxt(file_io.FileIO(path.decode(), 'r'),dtype='float32', delimiter=','), (1500, 100, 1))
        #print(features)
        return features, label

    def set_shape(features, label):
        features.set_shape([1500, 100, 1])
        label.set_shape([1])
        return features, label
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
    dataset = dataset.map(lambda features, label: set_shape(features, label))
    #if mode == tf.estimator.ModeKeys.TRAIN:
    #   dataset = dataset.shuffle(buffer_size = 10 * batch_size)\

    dataset = dataset.repeat(num_epochs).batch(batch_size)
    return dataset

def conv_lstm_model_function():

    # 30092513
    # inputs = tf.keras.layers.Input(shape=(1000,1))
    #
    # # a layer instance is callable on a tensor, and returns a tensor
    # x = tf.keras.layers.Conv1D(filters=32,
    #     kernel_size=16,
    #     padding="same")(inputs)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    # x = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)(x)
    # x = tf.keras.layers.Conv1D(filters=64,
    #     kernel_size=16,
    #     padding="same")(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    # x = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)(x)
    # x = tf.keras.layers.Conv1D(filters=128,
    #     kernel_size=16,
    #     padding="same")(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    # x = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)(x)
    # vector_rep = tf.keras.layers.Flatten()(x)
    # model = tf.keras.models.Model(inputs=inputs, outputs=vector_rep)
    #
    # inputs = tf.keras.layers.Input(shape=(150,1000,1))
    # x = tf.keras.layers.TimeDistributed(model)(inputs)
    # x = tf.keras.layers.LSTM(2048)(x)
    # x = tf.keras.layers.Dense(2048)(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    # x = tf.keras.layers.Dense(256)(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    # x = tf.keras.layers.Dense(128)(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    # predictions = tf.keras.layers.Dense(1)(x)

    # keras_large_conv_lstm_model_on_gpu
    # inputs = tf.keras.layers.Input(shape=(1000,1))
    #
    # # a layer instance is callable on a tensor, and returns a tensor
    # x = tf.keras.layers.Conv1D(filters=4,
    #     kernel_size=16,
    #     padding="same")(inputs)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    # x = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)(x)
    # x = tf.keras.layers.Conv1D(filters=8,
    #     kernel_size=16,
    #     padding="same")(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    # x = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)(x)
    # x = tf.keras.layers.Conv1D(filters=16,
    #     kernel_size=16,
    #     padding="same")(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    # x = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)(x)
    # vector_rep = tf.keras.layers.Flatten()(x)
    # model = tf.keras.models.Model(inputs=inputs, outputs=vector_rep)
    #
    # inputs = tf.keras.layers.Input(shape=(150,1000,1))
    # x = tf.keras.layers.TimeDistributed(model)(inputs)
    # x = tf.keras.layers.LSTM(128)(x)
    # x = tf.keras.layers.Dense(128)(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    # x = tf.keras.layers.Dense(64)(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    # predictions = tf.keras.layers.Dense(1)(x)


    # keras_1000000_conv_lstm_model
    with tf.device('/cpu:0'):
        inputs = tf.keras.layers.Input(shape=(100,1))

        # a layer instance is callable on a tensor, and returns a tensor
        x = tf.keras.layers.Conv1D(filters=16,
            kernel_size=10,
            padding="same")(inputs)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)(x)
        #x = tf.keras.layers.Permute((2, 1))(x)
        x = tf.keras.layers.CuDNNLSTM(32)(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)

        inputs = tf.keras.layers.Input(shape=(1500,100,1))
        x = tf.keras.layers.TimeDistributed(model)(inputs)
        x = tf.keras.layers.CuDNNLSTM(64, return_sequences=True)(x)
        x = tf.keras.layers.CuDNNLSTM(32)(x)
        x = tf.keras.layers.Dense(128)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        predictions = tf.keras.layers.Dense(1)(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=predictions)

    print(model.summary())
    parallel_model = tf.keras.utils.multi_gpu_model(model, gpus=4)
    parallel_model.compile(optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae', 'acc'])
    return parallel_model

    # keras 35000 conv lstm model
    # with tf.device('/cpu:0'):
    #     inputs = tf.keras.layers.Input(shape=(1000,1))
    #
    #     # a layer instance is callable on a tensor, and returns a tensor
    #     x = tf.keras.layers.Conv1D(filters=2,
    #         kernel_size=16,
    #         padding="same")(inputs)
    #     x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    #     x = tf.keras.layers.MaxPooling1D(pool_size=10, strides=10)(x)
    #     x = tf.keras.layers.Conv1D(filters=4,
    #         kernel_size=16,
    #         padding="same")(x)
    #     x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    #     x = tf.keras.layers.MaxPooling1D(pool_size=10, strides=10)(x)
    #     x = tf.keras.layers.Conv1D(filters=6,
    #         kernel_size=16,
    #         padding="same")(x)
    #     x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    #     x = tf.keras.layers.MaxPooling1D(pool_size=10, strides=10)(x)
    #     vector_rep = tf.keras.layers.Flatten()(x)
    #     model = tf.keras.models.Model(inputs=inputs, outputs=vector_rep)
    #
    #     inputs = tf.keras.layers.Input(shape=(150,1000,1))
    #     x = tf.keras.layers.TimeDistributed(model)(inputs)
    #     x = tf.keras.layers.CuDNNLSTM(64, return_sequences=True)(x)
    #     x = tf.keras.layers.CuDNNLSTM(32, return_sequences=True)(x)
    #     x = tf.keras.layers.CuDNNLSTM(16)(x)
    #     x = tf.keras.layers.Dense(16, activation='relu')(x)
    #     predictions = tf.keras.layers.Dense(1)(x)
    #     model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
    # #print(model.count_params())
    # #model = tf.keras.utils.multi_gpu_model(model, gpus=4)
    # model.compile(optimizer='adam',
    #     loss='mean_squared_error',
    #     metrics=['mae', 'acc'])
    # return model

    # keras_8000000_conv_lstm_model
    # with tf.device('/cpu:0'):
    #     inputs = tf.keras.layers.Input(shape=(1000,1))
    #
    #     # a layer instance is callable on a tensor, and returns a tensor
    #     x = tf.keras.layers.Conv1D(filters=2,
    #         kernel_size=16,
    #         padding="same")(inputs)
    #     x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    #     x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)
    #     x = tf.keras.layers.Conv1D(filters=4,
    #         kernel_size=16,
    #         padding="same")(x)
    #     x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    #     x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)
    #     x = tf.keras.layers.Conv1D(filters=6,
    #         kernel_size=16,
    #         padding="same")(x)
    #     x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    #     x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)
    #     vector_rep = tf.keras.layers.Flatten()(x)
    #     model = tf.keras.models.Model(inputs=inputs, outputs=vector_rep)
    #
    #     inputs = tf.keras.layers.Input(shape=(150,1000,1))
    #     x = tf.keras.layers.TimeDistributed(model)(inputs)
    #     x = tf.keras.layers.CuDNNLSTM(1024, return_sequences=True)(x)
    #     x = tf.keras.layers.CuDNNLSTM(512, return_sequences=True)(x)
    #     x = tf.keras.layers.CuDNNLSTM(256)(x)
    #     x = tf.keras.layers.Dense(256, activation='relu')(x)
    #     x = tf.keras.layers.Dropout(0.2)(x)
    #     x = tf.keras.layers.Dense(128, activation='relu')(x)
    #     x = tf.keras.layers.Dropout(0.2)(x)
    #     predictions = tf.keras.layers.Dense(1)(x)
    #     model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
    #
    # parallel_model = tf.keras.utils.multi_gpu_model(model, gpus=4)
    # parallel_model.compile(optimizer='adam',
    #     loss='mean_squared_error',
    #     metrics=['mae', 'acc'])
    # return parallel_model



def train_and_evaluate(output_dir):

    # EVAL_INTERVAL = 3600
    #
    # run_config = tf.estimator.RunConfig(save_summary_steps=10, save_checkpoints_secs = EVAL_INTERVAL,
    #                                   keep_checkpoint_max = 3)
    # signal_regressor = tf.keras.estimator.model_to_estimator(keras_model=conv_lstm_model_function(), model_dir=output_dir, config = run_config)

    eval_filename = 'eval.tar.gz'
    train_filename = 'train.tar.gz'

    subprocess.check_call(['gsutil', 'cp', os.path.join('gs://lanl-earthquake-gpu-large', eval_filename), eval_filename], stderr=sys.stdout)
    subprocess.check_call(['gsutil', 'cp', os.path.join('gs://lanl-earthquake-gpu-large', train_filename), train_filename], stderr=sys.stdout)

    subprocess.call(['tar', '-xf', eval_filename])

    subprocess.call(['tar', '-xf', train_filename])

    #tf.estimator.Estimator(
    #    model_fn=cnn_model_fn, model_dir=output_dir, config = run_config)

    training_dataset = read_dataset("train", num_epochs=30, mode=tf.estimator.ModeKeys.TRAIN, batch_size=BATCH_SIZE)

    # Pass a numpy array by using DataFrame.values
    validation_dataset = read_dataset("eval", num_epochs=1, mode=tf.estimator.ModeKeys.EVAL, batch_size=67)

    keras_model = conv_lstm_model_function()

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        os.path.join(JOB_DIR, 'keras_tensorboard'),
        histogram_freq=1)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=3)
    history = keras_model.fit(training_dataset,
                          epochs=30,
                          steps_per_epoch=int(TRAIN_EXAMPLES/BATCH_SIZE),
                          validation_data=validation_dataset,
                          validation_steps=25,
                          callbacks=[tensorboard_cb, early_stop],
                          verbose=1)
    export_path = tf.contrib.saved_model.save_keras_model(keras_model, JOB_DIR + '/keras_export')
    print("Model exported to: ", export_path)



def predict(output_dir):
    signal_regressor = tf.keras.estimator.model_to_estimator(keras_model=conv_lstm_model_function(), model_dir=output_dir)


    predictions = list()
    for i in signal_regressor.predict(lambda: read_dataset("./data/15000_processed_data/test",num_epochs=1,mode=tf.estimator.ModeKeys.PREDICT, batch_size=BATCH_SIZE)):
        predictions.append(i)
        print(i)



    with file_io.FileIO('predictions.pkl', 'wb') as f:
        pickle.dump(predictions, f)

    with file_io.FileIO('./data/15000_processed_data/file_names.pkl', 'rb') as f:
        file_list = pickle.load(f)

    df = pd.DataFrame(list(zip([i.split('/')[4][:-4] for i in file_list], [i[0] if i[0] < 0.0 else 0.0 for i in predictions])), columns=['seg_id', 'time_to_failure'])
    df['time_to_failure'] = df['time_to_failure'].apply(lambda x: 0.0 if x < 0.0 else x)
    df.to_csv('./data/15000_processed_data/submission_standard.csv', index=False)
