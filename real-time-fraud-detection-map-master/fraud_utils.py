# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:29:52 2019

@author: Lambert Rosique
@source : https://github.com/miguelgfierro/sciblog_support/tree/master/Intro_to_Fraud_Detection
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, log_loss, precision_score, recall_score
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
import requests

"""On initialise l’URL de l’api à l’adresse de notre serveur local
    et on stocke notre matrice dans le fichier lgb.model dans le répertoire save."""

# Constants
URL_API = 'http://localhost:5000'
BASELINE_MODEL = 'save/lgb.model'
FRAUD_THRESHOLD = 0.5

''' Les variables data_creditcard et data_worldcities sont initialisés grâce aux fichiers csv creditcard et worldcities. '''
dataset_creditcard = 'data/creditcard.csv'
dataset_worldcities = 'data/worldcities.csv'

''' Importation des données '''
# Dataset des fraudes
data_creditcard = pd.read_csv(dataset_creditcard)
# On importe les données géographiques des villes qu’on stocke dans la variable data_cities. 
data_cities = pd.read_csv(dataset_worldcities, usecols=['city_ascii', 'country', 'lat', 'lng'])
data_cities = data_cities.rename(columns={'city_ascii': 'city', 'lat': 'latitude', 'lng': 'longitude'})

#tirage aléatoire successif de ville grâce à la fonction sample.

def select_random_row_cities():
    global data_cities
    return data_cities.sample(n=1)
#tirage aléatoire successif de ville grâce à la fonction sample.

def select_random_row_creditcard():
    global data_creditcard
    return data_creditcard.sample(n=1)

#Division les données parmi X  et y. La taille du test est 0.2. 
#On initialise les données d’apprentissage et de test à partir de la division puis on les retourne.

def split_train_test(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

"""La fonction classification_metrics_binary compare les données y_true et  y_pred  pour définir :
-L ’accuracy c’est à dire le nombre de prédictions justes.
 - La matrice de confusion, constituée à partir des vrai négatifs, des faux négatifs puis des vrais positifs et des faux positifs.
- la précision calculée à partir du nombre de prédictions vraies positives divisés par le nombre de prédictions (vraies et fausses) positives.
- le recall calculé à partir du nombre du nombre de prédictions vraies positives divisé par lui même et le nombre de prédictions fausses négatives.
- Le f1 est calculé à partir de la formule 2*((precision*recall)/(precision+recall))
"""
def classification_metrics_binary(y_true, y_pred):

    m_acc = accuracy_score(y_true, y_pred)
    m_f1 = f1_score(y_true, y_pred)
    m_precision = precision_score(y_true, y_pred)
    m_recall = recall_score(y_true, y_pred)
    m_conf = confusion_matrix(y_true, y_pred)
    report = {'Accuracy': m_acc, 'Precision': m_precision,
              'Recall': m_recall, 'F1': m_f1, 'Confusion Matrix': m_conf}
    return report

"""on fait la meme chose que
 dans la précédente avec y_true et y_prob"""

def classification_metrics_binary_prob(y_true, y_prob):
   
    m_auc = roc_auc_score(y_true, y_prob)
    m_logloss = log_loss(y_true, y_prob)
    report = {'AUC': m_auc, 'Log loss': m_logloss}
    return report

#convertion des données en base binaire grâce au seuil qui vaut 0.5 , si c’est vrai la valeur de la donnée vaudra 1 sinon 0.
def binarize_prediction(y, threshold=0.5):
   
    y_pred = np.where(y > threshold, 1, 0)
    return y_pred

"""On constitue la matrice de confusion à l’aide de la bibliothèque matplotlib et de son composant pyplot. 
Le thresh est défini par la valeur maximale de la matrice de confusion divisé par 2. 
La bibliothèque initialise la matrice puis l’affiche. 
Si la donnée de la matrice c’est m est supérieur 
au thresh alors elle apparaîtra en blanc sinon en noire. 
"""

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
 
    cm_max = cm.max()
    cm_min = cm.min()
    if cm_min > 0:
        cm_min = 0
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_max = 1
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm_max / 2.
    plt.clim(cm_min, cm_max)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i,
                 round(cm[i, j], 3),  # round to 3 decimals if they are float
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

"""initialisation du temps d’attente aléatoire en utilisant 
minwait la borne minimale et maxwait la borne maximale"""

def wait_random_time(minwait,maxwait):
    r = random.randint(minwait*100,maxwait*100)
    time.sleep(r/100)
    
"""test si le serveur enregistré sur l’url qui est donnée en paramètres est lancé.
 Si c’est le cas on retourne true sinon on retourne faux."""
def test_server_online():
    try:
        requests.get(URL_API)
        return True
    except:
        return False
