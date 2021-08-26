# imports
from __future__ import print_function, division

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import tensorflow as tf
from collections import Counter
from IPython.display import display
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.utils import class_weight
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow import keras
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score, confusion_matrix
from keras import optimizers, regularizers, initializers
import re
import time
from keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
# from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

target_names = ["c0","c1","c2","c3","c4","c5","c6","c7","c8","c9"]

#get_ipython().system('nvidia-smi')


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_DEVICE_ORDER"]
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1"; 

tf.config.list_physical_devices(device_type='GPU')

image_size = (224, 224)
batch_size = 32

# ### Teste com Múltiplos Modelos

# compare standalone models for binary classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot


# get a list of models to evaluate
def get_models():
    models = dict()
    models['lr'] = LogisticRegression()
    models['knn'] = KNeighborsClassifier()
    models['cart'] = DecisionTreeClassifier()
    models['rf'] = RandomForestClassifier()
    models['gbc'] = GradientBoostingClassifier()
    models['xgb'] = XGBClassifier()
    models['lgb'] = LGBMClassifier()
    models['svm'] = SVC()
    models['bayes'] = GaussianNB()
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# ### Imagens de Validação
input_path='/data/Split80_20/Train'
input_path_valid='/data/Split80_20/Valid'

#Existe um chapter no fim do notebook com exemplos de data augmentation.
image_generator=ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input,
                                   rotation_range=20,\
                                   width_shift_range=0.15,\
                                   height_shift_range=0.15,\
                                   zoom_range=[0.8,1.2] ,\
                                   fill_mode="constant")




#,brightness_range= [0.6, 1.5],height_shift_range=0.15,width_shift_range=0.15,rotation_range=15  ,fill_mode="constant"      

#Using ImageDataGenerator to read images from directories
image_size = (224, 224)
batch_size = 32
# train_datagen   = ImageDataGenerator(f.keras.applications.inception_resnet_v2.preprocess_input)

train_generator = image_generator.flow_from_directory(input_path,
                                                      shuffle=True,
                                                      target_size=image_size, batch_size=batch_size,
                                                      class_mode='categorical', seed=42)

image_valid_generator=ImageDataGenerator()

validation_generator = image_valid_generator.flow_from_directory(
    input_path_valid, # same directory as training data
    shuffle=False,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    seed=42) # set as validation data


# ### Modelos
path = r"/data/best_model_AllTrainable_BATCHSIZE64_OPTMIZERAdam_LR0.0001.h5"
model_vgg16 = keras.models.load_model(path)
model_vgg16.summary()

model_efficientenetb0 = keras.models.load_model(r'/data/modeloefficientenetb076_novo_split_acc')
model_efficientenetb0.load_weights(r'/data/NewSplit/efficientnetB0/weights-improvement-B0-55-0.81.hdf5')
model_efficientenetb0.summary()


# ### Predição
input_path='/data/final_split/treino'
input_path_valid='/data/final_split/validacao'
classes = os.listdir(input_path_valid)
PATH = input_path_valid + '/' + classes[0]
PATH

os.listdir(input_path)

# ### EfficientNet
start = time.time()

# img_dir=DATA_PATH + '/test'
imagens = []
categorias_treino = []
categorias_valid = []
cat_int_treino= []
cat_int_valid= []
print('A')
images_treino = []
images_valid = []
#PATH = input_path_valid + '/' + classes[0]

for classe in classes:
    
    PATH = input_path + '/' + classe
    categorias_treino.append(int(re.findall(r'\d+', classe)[0]))
    
    for img in os.listdir(PATH):
        cat_int_treino.append(int(re.findall(r'\d+', classe)[0]))
        imagens.append(img)
        img = os.path.join(PATH, img).replace('\\', '/')
        img = image.load_img(img, target_size=image_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images_treino.append(img)

# stack up images list to pass for prediction
images_treino = np.vstack(images_treino)
preditct_model_efficientenetb0_treino = model_efficientenetb0.predict(images_treino, batch_size=50)

for classe in classes:
    
    PATH = input_path_valid + '/' + classe
    categorias_valid.append(int(re.findall(r'\d+', classe)[0]))
    
    for img in os.listdir(PATH):
        cat_int_valid.append(int(re.findall(r'\d+', classe)[0]))
        imagens.append(img)
        img = os.path.join(PATH, img).replace('\\', '/')
        img = image.load_img(img, target_size=image_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images_valid.append(img)

# stack up images list to pass for prediction
images_valid = np.vstack(images_valid)
preditct_model_efficientenetb0_valid = model_efficientenetb0.predict(images_valid, batch_size=50)

round(pd.DataFrame(preditct_model_efficientenetb0_treino).describe(), 5)

y_train_one_hot = tf.one_hot(cat_int_treino, depth=10)
y_valid_one_hot = tf.one_hot(cat_int_valid, depth=10)

model_efficientenetb0.evaluate(images_treino, y_train_one_hot, verbose=1)
model_efficientenetb0.evaluate(images_valid, y_valid_one_hot, verbose=1)

pd.DataFrame([{'Modelo': 'EfficientNet',
               'Base': 'Treino',
              'Log Loss': np.round(log_loss(cat_int_treino, preditct_model_efficientenetb0_treino), 2),
              'Acurácia Balanceada': np.round(balanced_accuracy_score(cat_int_treino, np.argmax(preditct_model_efficientenetb0_treino,axis=1))*100, 2),
              'RocAuc OVR': np.round(roc_auc_score(cat_int_treino, preditct_model_efficientenetb0_treino, average='weighted', multi_class = 'ovr')*100, 2),
              'RocAuc OVO': np.round(roc_auc_score(cat_int_treino, preditct_model_efficientenetb0_treino, average='weighted', multi_class = 'ovo')*100, 2)}])

pd.DataFrame([{'Modelo': 'EfficientNet',
               'Base': 'Validação',
              'Log Loss': np.round(log_loss(cat_int_valid, preditct_model_efficientenetb0_valid), 2),
              'Acurácia Balanceada': np.round(balanced_accuracy_score(cat_int_valid, np.argmax(preditct_model_efficientenetb0_valid,axis=1))*100, 2),
              'RocAuc OVR': np.round(roc_auc_score(cat_int_valid, preditct_model_efficientenetb0_valid, average='weighted', multi_class = 'ovr')*100, 2),
              'RocAuc OVO': np.round(roc_auc_score(cat_int_valid, preditct_model_efficientenetb0_valid, average='weighted', multi_class = 'ovo')*100, 2)}])



cm = confusion_matrix(cat_int_treino, np.argmax(preditct_model_efficientenetb0_treino,axis=1))
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
sns.color_palette("pastel")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)

cm = confusion_matrix(cat_int_valid, np.argmax(preditct_model_efficientenetb0_valid,axis=1))
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
sns.color_palette("pastel")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)


# ### VGG16

images_2_treino = []
images_2_valid = []

for classe in classes:
    
    PATH = input_path + '/' + classe
    
    for img in os.listdir(PATH):

        img = os.path.join(PATH, img).replace('\\', '/')
        img = image.load_img(img, target_size=image_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images_2_treino.append(preprocess_input(img))

# stack up images list to pass for prediction
images_2_treino = np.vstack(images_2_treino)
preditct_model_vgg16_treino = model_vgg16.predict(images_2_treino, batch_size=50)


for classe in classes:
    
    PATH = input_path_valid + '/' + classe
    
    for img in os.listdir(PATH):

        img = os.path.join(PATH, img).replace('\\', '/')
        img = image.load_img(img, target_size=image_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images_2_valid.append(preprocess_input(img))

# stack up images list to pass for prediction
images_2_valid = np.vstack(images_2_valid)
preditct_model_vgg16_valid = model_vgg16.predict(images_2_valid, batch_size=50)
pd.DataFrame(preditct_model_vgg16_treino).describe()
pd.DataFrame(preditct_model_vgg16_valid).describe()
model_vgg16.evaluate(images_2_treino, y_train_one_hot, verbose=1)
model_vgg16.evaluate(images_2_valid, y_valid_one_hot, verbose=1)
pd.DataFrame([{'Modelo': 'VGG16',
               'Base': 'Treino',
               'Log Loss': np.round(log_loss(cat_int_treino, preditct_model_vgg16_treino), 2),
               'Acurácia Balanceada': np.round(balanced_accuracy_score(cat_int_treino, np.argmax(preditct_model_vgg16_treino,axis=1))*100, 2),
               'RocAuc OVR': np.round(roc_auc_score(cat_int_treino, preditct_model_vgg16_treino, average='weighted', multi_class = 'ovr')*100, 2),
               'RocAuc OVO': np.round(roc_auc_score(cat_int_treino, preditct_model_vgg16_treino, average='weighted', multi_class = 'ovo')*100, 2)}])

pd.DataFrame([{'Modelo': 'VGG16',
               'Base': 'Validação',
               'Log Loss': np.round(log_loss(cat_int_valid, preditct_model_vgg16_valid), 2),
               'Acurácia Balanceada': np.round(balanced_accuracy_score(cat_int_valid, np.argmax(preditct_model_vgg16_valid,axis=1))*100, 2),
               'RocAuc OVR': np.round(roc_auc_score(cat_int_valid, preditct_model_vgg16_valid, average='weighted', multi_class = 'ovr')*100, 2),
               'RocAuc OVO': np.round(roc_auc_score(cat_int_valid, preditct_model_vgg16_valid, average='weighted', multi_class = 'ovo')*100, 2)}])

cm = confusion_matrix(cat_int_treino, np.argmax(preditct_model_vgg16_treino,axis=1))
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
sns.color_palette("pastel")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)

cm = confusion_matrix(cat_int_valid, np.argmax(preditct_model_vgg16_valid,axis=1))
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
sns.color_palette("pastel")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)
# # Ensemble

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    idea from this post:
    http://www.kaggle.com/c/emc-data-science/forums/t/2149/is-anyone-noticing-difference-betwen-validation-and-leaderboard-error/12209#post12209

    Parameters
    ----------
    y_true : array, shape = [n_samples]
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota


# ### Ensemble 1
# 
# $MODELO 1 = 0.5*VGG + 0.5 EfficientNet$
pred_ensemble1 = preditct_model_vgg16_treino*0.5 + preditct_model_efficientenetb0_treino*0.5
class_indexes = np.argmax(pred_ensemble1,axis=1)
round(pd.DataFrame(pred_ensemble1).describe(), 5)

pd.DataFrame([{'Modelo': 'Ensemble 1',
              'Log Loss': np.round(log_loss(cat_int_treino, pred_ensemble1), 2),
              'Acurácia Balanceada': np.round(balanced_accuracy_score(cat_int_treino, class_indexes)*100, 2),
              'RocAuc OVR': np.round(roc_auc_score(cat_int_treino, pred_ensemble1, average='weighted', multi_class = 'ovr')*100, 2),
              'RocAuc OVO': np.round(roc_auc_score(cat_int_treino, pred_ensemble1, average='weighted', multi_class = 'ovo')*100, 2)}])

cm = confusion_matrix(cat_int_treino, class_indexes)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
sns.color_palette("pastel")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)


# ### Ensemble 2
# $MODELO 2 = 0.3*VGG + 0.78EfficientNet$

pred_ensemble2 = preditct_model_vgg16_treino*0.3 + preditct_model_efficientenetb0_treino*0.7
class_indexes2 = np.argmax(pred_ensemble2,axis=1)
round(pd.DataFrame(pred_ensemble2).describe(), 5)

pd.DataFrame([{'Modelo': 'Ensemble 2',
              'Log Loss': np.round(log_loss(cat_int_treino, pred_ensemble2), 2),
              'Acurácia Balanceada': np.round(balanced_accuracy_score(cat_int_treino, class_indexes2)*100, 2),
              'RocAuc OVR': np.round(roc_auc_score(cat_int_treino, pred_ensemble2, average='weighted', multi_class = 'ovr')*100, 2),
              'RocAuc OVO': np.round(roc_auc_score(cat_int_treino, pred_ensemble2, average='weighted', multi_class = 'ovo')*100, 2)}])

cm = confusion_matrix(cat_int_treino, class_indexes2)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
sns.color_palette("pastel")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)


# ### Ensemble 3
# 
# $MODELO 3 = 0.7*VGG + 0.3*EfficientNet$

pred_ensemble3 = preditct_model_vgg16_treino*0.7 + preditct_model_efficientenetb0_treino*0.3
class_indexes3 = np.argmax(pred_ensemble3,axis=1)
round(pd.DataFrame(pred_ensemble3).describe(), 5)

pd.DataFrame([{'Modelo': 'Ensemble 3',
              'Log Loss': np.round(log_loss(cat_int_treino, pred_ensemble3), 2),
              'Acurácia Balanceada': np.round(balanced_accuracy_score(cat_int_treino, class_indexes3)*100, 2),
              'RocAuc OVR': np.round(roc_auc_score(cat_int_treino, pred_ensemble3, average='weighted', multi_class = 'ovr')*100, 2),
              'RocAuc OVO': np.round(roc_auc_score(cat_int_treino, pred_ensemble3, average='weighted', multi_class = 'ovo')*100, 2)}])

cm = confusion_matrix(cat_int_treino, class_indexes3)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
sns.color_palette("pastel")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)


# ### Escolha de Pesos
pesos = []

for numb in np.arange(0.01, 1, 0.01):
    predict_ensemble = preditct_model_vgg16_treino*numb + preditct_model_efficientenetb0_treino*(1-numb)
    class_index = np.argmax(predict_ensemble,axis=1)
    pesos.append({'Peso VGG16': numb, 'Peso EfficientNet': (1-numb),
              'Log Loss': np.round(log_loss(cat_int_treino, predict_ensemble), 4),
              'Acurácia Balanceada': np.round(balanced_accuracy_score(cat_int_treino, class_index)*100, 4),
              'RocAuc OVR': np.round(roc_auc_score(cat_int_treino, predict_ensemble, average='weighted', multi_class = 'ovr')*100, 4),
              'RocAuc OVO': np.round(roc_auc_score(cat_int_treino, predict_ensemble, average='weighted', multi_class = 'ovo')*100, 4)})

pd.DataFrame(pesos).head(50)
pd.DataFrame(pesos).tail(50)


# ### Ensemble 4
# 
# $MODELO 4 = 0.51*VGG + 0.49*EfficientNet$
pred_ensemble4_treino = preditct_model_vgg16_treino*0.51 + preditct_model_efficientenetb0_treino*0.49
class_indexes4_treino = np.argmax(pred_ensemble4_treino,axis=1)
round(pd.DataFrame(pred_ensemble4_treino).describe(), 5)

pred_ensemble4_valid = preditct_model_vgg16_valid*0.51 + preditct_model_efficientenetb0_valid*0.49
class_indexes4_valid = np.argmax(pred_ensemble4_valid,axis=1)
round(pd.DataFrame(pred_ensemble4_valid).describe(), 5)

pd.DataFrame([{'Modelo': 'Ensemble 4',
               'Base': 'Treino',
              'Log Loss': np.round(log_loss(cat_int_treino, pred_ensemble4_treino), 2),
              'Acurácia Balanceada': np.round(balanced_accuracy_score(cat_int_treino, class_indexes4_treino)*100, 2),
              'RocAuc OVR': np.round(roc_auc_score(cat_int_treino, pred_ensemble4_treino, average='weighted', multi_class = 'ovr')*100, 2),
              'RocAuc OVO': np.round(roc_auc_score(cat_int_treino, pred_ensemble4_treino, average='weighted', multi_class = 'ovo')*100, 2)}])

pd.DataFrame([{'Modelo': 'Ensemble 4',
               'Base': 'Validação',
              'Log Loss': np.round(log_loss(cat_int_valid, pred_ensemble4_valid), 2),
              'Acurácia Balanceada': np.round(balanced_accuracy_score(cat_int_valid, class_indexes4_valid)*100, 2),
              'RocAuc OVR': np.round(roc_auc_score(cat_int_valid, pred_ensemble4_valid, average='weighted', multi_class = 'ovr')*100, 2),
              'RocAuc OVO': np.round(roc_auc_score(cat_int_valid, pred_ensemble4_valid, average='weighted', multi_class = 'ovo')*100, 2)}])

cm = confusion_matrix(cat_int_treino, class_indexes4_treino)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
sns.color_palette("pastel")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)

cm = confusion_matrix(cat_int_valid, class_indexes4_valid)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
sns.color_palette("pastel")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)


# ### Testando outras formas de Ensemble
# 
# Testar três tipos de modelos distintos ao fazermos o ensemble
# 
# * Regressão Logística
# * Floresta Aleatória
# * Gradient Boosting

# ### Regressão Logística

X = pred_ensemble4_treino
y = cat_int_treino

logreg = LogisticRegression()

# Create an instance of Logistic Regression Classifier and fit the data.
logreg.fit(X, y)

pred_ensemble4_lr = logreg.predict_proba(pred_ensemble4_valid)
class_indexes_lr = np.argmax(pred_ensemble4_lr,axis=1)
round(pd.DataFrame(pred_ensemble4_lr).describe(), 5)

pd.DataFrame([{'Modelo': 'Ensemble 4',
              'Log Loss': np.round(log_loss(cat_int_valid, pred_ensemble4_lr), 2),
              'Acurácia Balanceada': np.round(balanced_accuracy_score(cat_int_valid, class_indexes_lr)*100, 2),
              'RocAuc OVR': np.round(roc_auc_score(cat_int_valid, pred_ensemble4_lr, average='weighted', multi_class = 'ovr')*100, 2),
              'RocAuc OVO': np.round(roc_auc_score(cat_int_valid, pred_ensemble4_lr, average='weighted', multi_class = 'ovo')*100, 2)}])

cm = confusion_matrix(cat_int_valid, class_indexes_lr)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
sns.color_palette("pastel")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgbm = LGBMClassifier(subsample=0.8)

# Create an instance of Logistic Regression Classifier and fit the data.
lgbm.fit(X_train, y_train)

pred_ensemble4_lgbm = lgbm.predict_proba(X_test)
class_indexes_lgbm = np.argmax(pred_ensemble4_lgbm,axis=1)
round(pd.DataFrame(pred_ensemble4_lgbm).describe(), 5)

pd.DataFrame([{'Modelo': 'Ensemble LGBM',
               'Base': 'Treino',
              'Log Loss': np.round(log_loss(y_test, pred_ensemble4_lgbm), 2),
              'Acurácia Balanceada': np.round(balanced_accuracy_score(y_test, class_indexes_lgbm)*100, 2),
              'RocAuc OVR': np.round(roc_auc_score(y_test, pred_ensemble4_lgbm, average='weighted', multi_class = 'ovr')*100, 2),
              'RocAuc OVO': np.round(roc_auc_score(y_test, pred_ensemble4_lgbm, average='weighted', multi_class = 'ovo')*100, 2)}])

pred_ensemble4_lgbm_valid = lgbm.predict_proba(pred_ensemble4_valid)
class_indexes_lgbm_valid = np.argmax(pred_ensemble4_lgbm_valid,axis=1)
round(pd.DataFrame(pred_ensemble4_lgbm_valid).describe(), 5)

pd.DataFrame([{'Modelo': 'Ensemble LGBM',
               'Base': 'Validação',
              'Log Loss': np.round(log_loss(cat_int_valid, pred_ensemble4_lgbm_valid), 2),
              'Acurácia Balanceada': np.round(balanced_accuracy_score(cat_int_valid, class_indexes_lgbm_valid)*100, 2),
              'RocAuc OVR': np.round(roc_auc_score(cat_int_valid, pred_ensemble4_lgbm_valid, average='weighted', multi_class = 'ovr')*100, 2),
              'RocAuc OVO': np.round(roc_auc_score(cat_int_valid, pred_ensemble4_lgbm_valid, average='weighted', multi_class = 'ovo')*100, 2)}])


# #### Fazendo só com a base de validação

X = pred_ensemble4_valid
y = cat_int_valid

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgbm = LGBMClassifier()

# Create an instance of Logistic Regression Classifier and fit the data.
lgbm.fit(X_train, y_train)

pred_ensemble4_lgbm = lgbm.predict_proba(X_test)
class_indexes_lgbm = np.argmax(pred_ensemble4_lgbm,axis=1)
round(pd.DataFrame(pred_ensemble4_lgbm).describe(), 5)

pd.DataFrame([{'Modelo': 'Ensemble LGBM',
               'Base': 'Validação',
              'Log Loss': np.round(log_loss(y_test, pred_ensemble4_lgbm), 2),
              'Acurácia Balanceada': np.round(balanced_accuracy_score(y_test, class_indexes_lgbm)*100, 2),
              'RocAuc OVR': np.round(roc_auc_score(y_test, pred_ensemble4_lgbm, average='weighted', multi_class = 'ovr')*100, 2),
              'RocAuc OVO': np.round(roc_auc_score(y_test, pred_ensemble4_lgbm, average='weighted', multi_class = 'ovo')*100, 2)}])


# X = pred_ensemble4_treino
# y = cat_int_treino

pred_ensemble4_lgbm_treino = lgbm.predict_proba(pred_ensemble4_treino)
class_indexes_lgbm_treino = np.argmax(pred_ensemble4_lgbm_treino,axis=1)
round(pd.DataFrame(pred_ensemble4_lgbm_treino).describe(), 5)

pd.DataFrame([{'Modelo': 'Ensemble LGBM',
               'Base': 'Treino',
              'Log Loss': np.round(log_loss(cat_int_treino, pred_ensemble4_lgbm_treino), 2),
              'Acurácia Balanceada': np.round(balanced_accuracy_score(cat_int_treino, class_indexes_lgbm_treino)*100, 2),
              'RocAuc OVR': np.round(roc_auc_score(cat_int_treino, pred_ensemble4_lgbm_treino, average='weighted', multi_class = 'ovr')*100, 2),
              'RocAuc OVO': np.round(roc_auc_score(cat_int_treino, pred_ensemble4_lgbm_treino, average='weighted', multi_class = 'ovo')*100, 2)}])


# ### Achando os melhoras parâmetros do LightGBM

def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3


param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

fit_params={"early_stopping_rounds":30,
            "eval_metric" : 'multi_logloss', 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': 100,
            'categorical_feature': 'auto'}

#This parameter defines the number of HP points to be tested
n_HP_points_to_test = 100

#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
clf = lgb.LGBMClassifier(max_depth=-1, num_classes=10, objective = 'multi:softmax', random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test, 
    n_iter=n_HP_points_to_test,
    scoring='balanced_accuracy',
    cv=3,
    refit=True,
    random_state=314,
    verbose=True)

gs.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))

opt_parameters = gs.best_params_

clf_sw = lgb.LGBMClassifier(**clf.get_params())
#set optimal parameters
clf_sw.set_params(**opt_parameters)
clf_sw.fit(X_train, y_train)
pred_ensemble4_lgbm_clf_sw = clf_sw.predict_proba(X_test)
class_indexes_lgbm_clf_sw = np.argmax(pred_ensemble4_lgbm_clf_sw,axis=1)
round(pd.DataFrame(pred_ensemble4_lgbm_clf_sw).describe(), 5)

pd.DataFrame([{'Modelo': 'Ensemble LGBM',
               'Base': 'Treino',
              'Log Loss': np.round(log_loss(y_test, pred_ensemble4_lgbm_clf_sw), 2),
              'Acurácia Balanceada': np.round(balanced_accuracy_score(y_test, class_indexes_lgbm_clf_sw)*100, 2),
              'RocAuc OVR': np.round(roc_auc_score(y_test, pred_ensemble4_lgbm_clf_sw, average='weighted', multi_class = 'ovr')*100, 2),
              'RocAuc OVO': np.round(roc_auc_score(y_test, pred_ensemble4_lgbm_clf_sw, average='weighted', multi_class = 'ovo')*100, 2)}])

log_loss(y_test, pred_ensemble4_lgbm_clf_sw)

balanced_accuracy_score(y_test, class_indexes_lgbm_clf_sw)

pred_ensemble4_lgbm_clf_sw = clf_sw.predict_proba(pred_ensemble4_valid)
class_indexes_lgbm_clf_sw = np.argmax(pred_ensemble4_lgbm_clf_sw,axis=1)
round(pd.DataFrame(pred_ensemble4_lgbm_clf_sw).describe(), 5)

cm = confusion_matrix(y_test, class_indexes_lgbm_clf_sw)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
sns.color_palette("pastel")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)
