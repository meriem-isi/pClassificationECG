from numpy import save
import csv
from keras.models import load_model
from scipy import io
import help1 as pc
import ecg_plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Add
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPool1D, ZeroPadding1D, LSTM, Bidirectional
from keras.models import Sequential, Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.layers.merge import concatenate
from scipy import optimize
from scipy.io import loadmat
import os

labels, ecg_filenames = pc.import_key_data("Training_2/")
ecg_filenames = np.asarray(ecg_filenames)
SNOMED_scored=pd.read_csv("physionet-snomed-mappings/SNOMED_mappings_scored.csv", sep=";")

SNOMED_unscored=pd.read_csv("physionet-snomed-mappings/SNOMED_mappings_unscored.csv", sep=";")
df_labels = pc.make_undefined_class(labels,SNOMED_unscored)
y , snomed_classes = pc.onehot_encode(df_labels)
y_all_comb = pc.get_labels_for_all_combinations(y)
print("Total number of unique combinations of diagnosis+:+ {}".format(len(np.unique(y_all_comb))))
folds = pc.split_data(labels, y_all_comb)

order_array = folds[0][0]


def shuffle_batch_generator(batch_size, gen_x, gen_y):
    np.random.shuffle(order_array)
    batch_features = np.zeros((batch_size, 5000, 12))
    batch_labels = np.zeros((batch_size, snomed_classes.shape[0]))  # drop undef class
    while True:
        for i in range(batch_size):
            #next(input):générer les batches aléatoires
            batch_features[i] = next(gen_x)
            batch_labels[i] = next(gen_y)
       # retour batch input ,batch targets
        yield batch_features, batch_labels


def generate_y_shuffle(y_train):
    while True:
        for i in order_array:
            y_shuffled = y_train[i]
            yield y_shuffled


def generate_X_shuffle(X_train):
    while True:
        for i in order_array:
            # if filepath.endswith(".mat"):
            data, header_data = pc.load_challenge_data(X_train[i])
            X_train_new = pad_sequences(data, maxlen=5000, truncating='post', padding="post")
            X_train_new = X_train_new.reshape(5000, 12)
            yield X_train_new
###################################################################################################################################

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_AUC', factor=0.1, patience=1, verbose=1, mode='max',
    min_delta=0.0001, cooldown=0, min_lr=0
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', mode='max', verbose=1, patience=2)


def thr_chall_metrics(thr, label, output_prob):
    return -pc.compute_challenge_metric_for_opt(label, np.array(output_prob > thr))


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file, 'r') as f:
        header_data = f.readlines()
    return data, header_data


def generate_validation_data(ecg_filenames, y, test_order_array):
    y_train_gridsearch = y[test_order_array]
    ecg_filenames_train_gridsearch = ecg_filenames[test_order_array]

    ecg_train_timeseries = []
    for names in ecg_filenames_train_gridsearch:
        data, header_data = load_challenge_data(names)
        data = pad_sequences(data, maxlen=5000, truncating='post', padding="post")
        ecg_train_timeseries.append(data)
    X_train_gridsearch = np.asarray(ecg_train_timeseries)

    X_train_gridsearch = X_train_gridsearch.reshape(ecg_filenames_train_gridsearch.shape[0], 5000, 12)

    return X_train_gridsearch, y_train_gridsearch
####################################################################################################""""

def compute_modified_confusion_matrix(labels, outputs):

    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0 / normalization

    return A


def plot_normalized_conf_matrix_dev(y_pred, ecg_filenames, y, val_fold, threshold, snomedclasses):
    df_cm = pd.DataFrame(compute_modified_confusion_matrix(generate_validation_data(ecg_filenames, y, val_fold)[1],
                                                           (y_pred > threshold) * 1), columns=snomedclasses,
                         index=snomedclasses)
    df_cm = df_cm.fillna(0)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    df_norm_col = (df_cm - df_cm.mean()) / df_cm.std()
    plt.figure(figsize=(36, 14))
    sns.set(font_scale=1.4)
    sns.heatmap(df_norm_col, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt=".2f", cbar=False)  # font size

###############################################################################################

lenet_5_model=Sequential()

lenet_5_model.add(Conv1D(filters=6, kernel_size=3, padding='same', input_shape=(5000,12)))
lenet_5_model.add(BatchNormalization())
lenet_5_model.add(Activation('relu'))
lenet_5_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
lenet_5_model.add(Conv1D(filters=16, strides=1, kernel_size=5))
lenet_5_model.add(BatchNormalization())
lenet_5_model.add(Activation('relu'))
lenet_5_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))

lenet_5_model.add(GlobalAveragePooling1D())


lenet_5_model.add(Dense(64, activation='relu'))

lenet_5_model.add(Dense(32, activation='relu'))

lenet_5_model.add(Dense(27, activation = 'sigmoid'))

lenet_5_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.BinaryAccuracy(
        name='accuracy', dtype=None, threshold=0.5),tf.keras.metrics.Recall(name='Recall'),tf.keras.metrics.Precision(name='Precision'),
                    tf.keras.metrics.AUC(
        num_thresholds=200,
        curve="ROC",
        summation_method="interpolation",
        name="AUC",
        dtype=None,
        thresholds=None,
        multi_label=True,
        label_weights=None,
    )])
lenet_5_model.summary()

batchsize = 10

#lenet_5_model.fit(x=shuffle_batch_generator(batch_size=batchsize, gen_x=generate_X_shuffle(ecg_filenames), gen_y=generate_y_shuffle(y)), epochs=10, steps_per_epoch=(len(order_array)/(batchsize*10)), validation_data=pc.generate_validation_data(ecg_filenames,y,folds[0][1]), callbacks=[reduce_lr,early_stop])

#new_weights=pc.calculating_class_weights(y)

#keys = np.arange(0,27,1)
#weight_dictionary = dict(zip(keys, new_weights.T[1]))

lenet_5_model.save('models/cnnModel.h5')
lenet_5_model.save_weights("models/cnn_weightsModel.h5")

#y_pred = lenet_5_model.predict(x=pc.generate_validation_data(ecg_filenames,y,folds[0][1])[0])
#init_thresholds = np.arange(0,1,0.05)

#all_scores = pc.iterate_threshold(y_pred, ecg_filenames, y ,folds[0][1] )


#new_best_thr = optimize.fmin(thr_chall_metrics, args=(pc.generate_validation_data(ecg_filenames,y,folds[0][1])[1],y_pred), x0=init_thresholds[all_scores.argmax()]*np.ones(27))
#plot_normalized_conf_matrix_dev(y_pred, ecg_filenames, y, folds[0][1], new_best_thr, snomed_classes)
#plt.savefig("confusion_matrix_lenet5.png", dpi=100)

