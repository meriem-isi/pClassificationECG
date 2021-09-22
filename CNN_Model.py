from numpy import save
import csv
#from keras.models import load_model
from scipy import io
#import help1 as pc
#import ecg_plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Add
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPool1D, ZeroPadding1D, LSTM, Bidirectional
from tensorflow.keras.models import Sequential, Model

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#from tensorflow.keras.layers.merge import concatenate
from scipy import optimize
from scipy.io import loadmat
import os
from scipy import signal
import tqdm

# import all filenames, labels, gender and age from the files in the directory
def import_key_data(path):
  gender=[]
  age=[]
  labels=[]
  ecg_filenames=[]
  for ecgfilename in tqdm.tqdm(sorted(os.listdir(path))):
      if ecgfilename.endswith(".mat"):
          data, header_data = load_challenge_data(path+ecgfilename)
          labels.append(header_data[15][5:-1])
          ecg_filenames.append(path+ecgfilename)
          gender.append(header_data[14][6:-1])
          age.append(header_data[13][6:-1])
  return gender, age, labels, ecg_filenames

def load_header(header_file):
    with open(header_file, 'r') as f:
        header = f.read()
    return header

def get_labels(header):
    labels = list()
    for l in header.split('\n'):
        if l.startswith('#Dx'):
            try:
                entries = l.split(': ')[1].split(',')
                for entry in entries:
                    labels.append(entry.strip())
            except:
                pass
    return labels
    
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False
        
def is_number(x):
  try:
      float(x)
      return True
  except (ValueError, TypeError):
      return False

# Split data using stratified CV to keep the same distribution in the training and test split
def split_data(labels, y_all_comb):
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(labels,y_all_comb))
    print("Training split: {}".format(len(folds[0][0])))
    print("Validation split: {}".format(len(folds[0][1])))
    return folds

def get_labels_for_all_combinations(y):
    y_all_combinations = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
    return y_all_combinations

def shuffle_batch_generator(batch_size, gen_x, gen_y, num_classes):
    # shuffle the order of the input data - order_array is a global variable
    np.random.shuffle(order_array)
    # An empty array for the ECGs
    batch_features = np.zeros((batch_size, 5000, 12))'
    # An empty array for the labels
    batch_labels = np.zeros((batch_size, num_classes))  # drop undef class
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
            data, header_data = load_challenge_data(X_train[i])
            # if the sample frequenzy is not equal to 500Hz -> do up or down sampling to 500Hz
            if int(header_data[0].split(" ")[2]) != 500:
              data_new = np.ones([12,int((int(header_data[0].split(" ")[3])/int(header_data[0].split(" ")[2]))*500)])
              for i,j in enumerate(data):
                  data_new[i] = signal.resample(j, int((int(header_data[0].split(" ")[3])/int(header_data[0].split(" ")[2]))*500))
              data = data_new
            # Pad or truncate the 500Hz signal to 5000 samples (equal to 10 sec)
            X_train_new = pad_sequences(data, maxlen=5000, truncating='post', padding="post")
            # reshape to make the ECG signal fit into the model
            X_train_new = X_train_new.reshape(5000, 12)
            yield X_train_new
###################################################################################################################################

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.1, patience=1, verbose=1, mode='max',
    min_delta=0.0001, cooldown=0, min_lr=0
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=2)

def compute_accuracy(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    num_correct_recordings = 0
    for i in range(num_recordings):
        if np.all(labels[i, :]==outputs[i, :]):
            num_correct_recordings += 1

    return float(num_correct_recordings) / float(num_recordings)

# A prediction threshold optimization algorithm - often used in imbalanced multi-label classifications 
def thr_chall_metrics(thr, label, output_prob):
    return -compute_accuracy(labels, (output_prob >= thr))

# Load ECG and meta data using filename
def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file, 'r') as f:
        header_data = f.readlines()
    return data, header_data


def generate_validation_data(ecg_filenames, y, test_order_array):
    # add labels for test data to y_val
    y_val=y[test_order_array]
    # add ECG filenames for test data to ecg_filenames_val
    ecg_filenames_val=ecg_filenames[test_order_array]
    # make a list to store all loaded test ECGs
    all_ecgs=[]
    for names in ecg_filenames_val:
        # load ECG into data and meta data to header_data
        data,header_data= load_challenge_data(names)
        # if the sample frequenzy is not equal to 500Hz -> do up or down sampling to 500Hz
        if int(header_data[0].split(" ")[2]) != 500:
            data_new = np.ones([12,int((int(header_data[0].split(" ")[3])/int(header_data[0].split(" ")[2]))*500)])
            for i,j in enumerate(data):
                data_new[i] = signal.resample(j, int((int(header_data[0].split(" ")[3])/int(header_data[0].split(" ")[2]))*500))
            data = data_new
        # Pad or truncate the 500Hz signal to 5000 samples (equal to 10 sec)
        data = pad_sequences(data, maxlen=5000, truncating='post',padding="post")
        # reshape to make the ECG signal fit into the model
        data = data.reshape(data.shape[1],data.shape[0])
        all_ecgs.append(data)
    all_ecgs = np.asarray(all_ecgs)

    return all_ecgs, y_val


def pred_batch_generator(batch_size, gen_x): 
    batch_features = np.zeros((batch_size,5000, 12))
    while True:
        for i in range(batch_size):
            batch_features[i] = next(gen_x)
        yield batch_features   


def generate_X_pred(X_train_file, val_index):
    while True:
        for i in val_index:
          # load ECG into data and meta data to header_data
          data, header_data = load_challenge_data(X_train_file[i])
          # if the sample frequenzy is not equal to 500Hz -> do up or down sampling to 500Hz
          if int(header_data[0].split(" ")[2]) != 500:
              data_new = np.ones([12,int((int(header_data[0].split(" ")[3])/int(header_data[0].split(" ")[2]))*500)])
              for i,j in enumerate(data):
                  data_new[i] = signal.resample(j, int((int(header_data[0].split(" ")[3])/int(header_data[0].split(" ")[2]))*500))
              data = data_new
          # Pad or truncate the 500Hz signal to 5000 samples (equal to 10 sec)
          data = pad_sequences(data, maxlen=5000, truncating='post',padding="post")
          # reshape to make the ECG signal fit into the model
          data = data.reshape(data.shape[1],data.shape[0])
          yield data
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

print("type in your data path:")
path1 = input()
print("type in the path to the SNOMED files")
path2 = input()
gender, age, labels, ecg_filenames = import_key_data(path1)
ecg_filenames = np.asarray(ecg_filenames)

classes = set()
for ecg_file in tqdm.tqdm(ecg_filenames):
  header_file = ecg_file.replace('.mat','.hea')
  header = load_header(header_file)
  classes |= set(get_labels(header))
if all(is_integer(x) for x in classes):
    classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
else:
    classes = sorted(classes) # Sort classes alphanumerically if not numbers.
num_classes = len(classes)


SNOMED_scored=pd.read_csv(path2, sep=",")
lab_arr = np.asarray(SNOMED_scored['SNOMED CT Code'], dtype="str")
scored_classes = []
for i in classes:
  for j in lab_arr:
    if i == '':
      continue
    if i == j:
      scored_classes.append(i)
scored_classes = sorted(scored_classes)


num_recordings = len(ecg_filenames)
num_classes = len(scored_classes)
labels = np.zeros((num_recordings, num_classes), dtype=np.bool) # One-hot encoding of classes

for i in tqdm.tqdm(range(len(ecg_filenames))):
  current_labels = get_labels(load_header(ecg_filenames[i].replace('.mat','.hea')))
  for label in current_labels:
      if label in scored_classes:
          j = scored_classes.index(label)
          labels[i, j] = 1
labels = labels *1
labels = np.asarray(labels)

y_all_comb = get_labels_for_all_combinations(labels)
print("Total number of unique combinations of diagnosis+:+ {}".format(len(np.unique(y_all_comb))))

folds = split_data(labels, y_all_comb)

#df_labels = pc.make_undefined_class(labels,SNOMED_unscored)
#y , snomed_classes = pc.onehot_encode(df_labels)
#y_all_comb = pc.get_labels_for_all_combinations(y)

#folds = pc.split_data(labels, y_all_comb)

order_array = folds[0][0]


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

lenet_5_model.add(Dense(num_classes, activation = 'sigmoid'))

lenet_5_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.BinaryAccuracy(
        name='accuracy', dtype=None, threshold=0.5)])
lenet_5_model.summary()

batchsize = 10

lenet_5_model.fit(x=shuffle_batch_generator(batch_size=batchsize, gen_x=generate_X_shuffle(ecg_filenames), gen_y=generate_y_shuffle(labels), num_classes=num_classes), epochs=10, steps_per_epoch=(len(order_array)/(batchsize*10)), 
#validation_data=generate_validation_data(ecg_filenames,labels,folds[0][1]), callbacks=[reduce_lr,early_stop]
)
y_pred = lenet_5_model.predict(generate_validation_data(ecg_filenames,labels,folds[0][1])[0])
print(f"Shape pred = {y_pred.shape}")
print(f"Shape GT = {labels.shape}")
#y_pred = lenet_5_model.predict_generator(pred_batch_generator(batch_size=batchsize, gen_x=generate_X_pred(ecg_filenames, folds[i][1])),steps=(len(folds[i][1])/batchsize))
print(f"Accuracy = {compute_accuracy(labels[folds[0][1]],((y_pred >= 0.5) *1))}")

lenet_5_model.save('models/cnnModel.h5')
lenet_5_model.save_weights("models/cnn_weightsModel.h5")


#init_thresholds = np.arange(0,1,0.05)

#all_scores = pc.iterate_threshold(y_pred, ecg_filenames, y ,folds[0][1] )


#new_best_thr = optimize.fmin(thr_chall_metrics, args=(pc.generate_validation_data(ecg_filenames,y,folds[0][1])[1],y_pred), x0=init_thresholds[all_scores.argmax()]*np.ones(27))
#plot_normalized_conf_matrix_dev(y_pred, ecg_filenames, y, folds[0][1], new_best_thr, snomed_classes)
#plt.savefig("confusion_matrix_lenet5.png", dpi=100)

