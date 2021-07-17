import pandas as pd
import numpy as np
from tensorflow import keras
from scipy import optimize
from scipy.io import loadmat
import pandas as pd
from keras.models import load_model
import CNN_Model as p
import help1 as pc
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

#################################################################################################"
SNOMED_scored=pd.read_csv("physionet-snomed-mappings/SNOMED_mappings_scored.csv", sep=";")
SNOMED_unscored=pd.read_csv("physionet-snomed-mappings/SNOMED_mappings_unscored.csv", sep=";")
###########################################################################################################
CNN_model=load_model("models/cnnModel.h5")

####################################################################################################
snomed_class_names=["Pacing Rhythm", "Prolonged QT Interval","Atrial Fibrillation","Atrial Flutter",
                 "Left Bundle Branch Block","Q Wave Abnormal","T Wave Abnormal","Prolonged PR Interval","Ventricular Premature Beats",
"Low QRS Voltages","1st Degree AV Block","Premature Atrial Contraction","Left Axis Deviation",
"Sinus Bradycardia","Bradycardia","Sinus Rhythm","Sinus Tachycardia","Premature Ventricular Contractions",
"Sinus Arrhythmia","Left Anterior Fascicular Block","Right Axis Deviation","Right Bundle Branch Block","T Wave Inversion",
"Supraventricular Premature Beats","Nonspecific Intraventricular Conduction Disorder","Incomplete Right Bundle Branch Block",
"Complete Right Bundle Branch Block"]
##########################################################################################################

#test my data  I have 3 value of three leads and I calculate the others 3  leads 

with open('1616172436008.txt', 'r') as f:
    data = f.readlines() # read raw lines into an array

cleaned_matrix = []
for raw_line in data:
    split_line = raw_line.strip().split(",")# ["1", "0" ... ]
    split_line = raw_line.strip().split(",")  # ["1", "0" ... ]
    nums_ls = [int(x.replace('"', '')) for x in split_line]  # get rid of the quotation marks and convert to int
    cleaned_matrix.append(nums_ls)

#delete line time ,in2 v1
#cleaned_matrix = np.delete(cleaned_matrix, 0, axis=0)
#delete colonne time
cleaned_matrix = np.delete(cleaned_matrix, 0, axis=1)
print(cleaned_matrix.shape)
m=np.zeros((3333,9))
cleaned_matrix=np.c_[cleaned_matrix,m]

print('shape')
print(cleaned_matrix.shape)
'''shape(3333, 12)'''

for i in range(len(cleaned_matrix)):
 cleaned_matrix[i][6] =float(cleaned_matrix[i][2])
#lead3 :lead2-lead 1 done
 cleaned_matrix[i][2] =float(cleaned_matrix[i][1]) - float(cleaned_matrix[i][0])
#lead aVF :lead2-lead *0.5
 cleaned_matrix[i][3] =  float(cleaned_matrix[i][1])- (float(cleaned_matrix[i][0])*(1/2))
#aVR :-0.5 *lead1 - lead2  *0.5 done
 cleaned_matrix[i][4] = -(0.5*float(cleaned_matrix[i][0] ))-(float(cleaned_matrix[i][1])*0.5)
#lead aVL :lead1 - lead 2 *0.5 done
 cleaned_matrix[i][5] = float(cleaned_matrix[i][0]) -(float(cleaned_matrix[i][1])*0.5)

y2=np.zeros((1667, 12))
cleaned_matrix=np.append(cleaned_matrix, y2, axis=0)
#print(cleaned_matrix.shape)
#(5000 ,12)
cleaned_matrix = {"val": cleaned_matrix}
x=cleaned_matrix['val'].reshape(1,cleaned_matrix['val'].shape[1],cleaned_matrix['val'].shape[0])
#xTest = tf.expand_dims(x, axis=-1)
X_train_new = pad_sequences(x, maxlen=5000, truncating='post', padding="post")
X_train_new = X_train_new.reshape(5000, 12)
yhat=CNN_model.predict(X_train_new)
print('yhat')
print(yhat)
################################################################################################################################
#test one file prediction different from actual
header_file="Training_2/A0645.hea
mat_file="Training_2/A0645.mat"
with open(header_file, 'r') as the_file:
    all_data = [line.strip() for line in the_file.readlines()]
    data = all_data[8:]
snomed_number=int(data[7][5:14])
value_unscored=SNOMED_unscored["Dx"][SNOMED_unscored["SNOMED CT Code"]==snomed_number].values
value_scored=SNOMED_scored["Dx"][SNOMED_scored["SNOMED CT Code"]==snomed_number].values
try:
    disease_unscored=value_unscored[0]
except:
    disease_unscored=""

try:
    disease_scored=value_scored[0]
except:
    disease_scored=""


yhat=CNN_model.predict(x=loadmat(mat_file)['val'].reshape(1,loadmat(mat_file)['val'].shape[1],loadmat(mat_file)['val'].shape[0]))
print("Predicted: "+snomed_class_names[np.argmax(yhat)])
if disease_unscored!="":
    print("Actual: "+disease_unscored)
else:
    print("Actual: "+disease_scored)

################################################

y_pred = CNN_model.predict(x=p.generate_validation_data(p.ecg_filenames,p.y,p.folds[0][1])[0])
print('y_pred')
print(y_pred)
init_thresholds = np.arange(0,1,0.05)

all_scores = pc.iterate_threshold(y_pred, p.ecg_filenames, p.y ,p.folds[0][1] )
print('fin scores')

#Optimisation des seuils

new_best_thr = optimize.fmin(p.thr_chall_metrics, args=(pc.generate_validation_data(p.ecg_filenames,p.y,p.folds[0][1])[1],y_pred), x0=init_thresholds[all_scores.argmax()]*np.ones(27))
print('new_best_thr')
print(new_best_thr)

