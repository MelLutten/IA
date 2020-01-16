# -*- coding: utf-8 -*-
'''
Created on Mon Jul 22 18:29:52 2019

@author: Lambert Rosique
'''

import json
import lightgbm as lgb
import fraud_utils as fu
import requests
import time

''' Modèle LGB '''
def train_model_lgb():
    #X_train et Y_train sont des variables d’apprentissages et X_test et Y_test sont des variables de tests.
    #X_train X_test X_train  Y_test prend les valeurs des données  du fichier credit card.

    global X_train, X_test, y_train, y_test
    # Découpage des données en train/test sets
    X_train, X_test, y_train, y_test = fu.split_train_test(fu.data_creditcard.drop('Class', axis=1), fu.data_creditcard['Class'], test_size=0.2)
    
    # Datasets
    #lgb_train  est initialisé à  partir de variables X_train, y_train
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    
    
    # Parameters
'''
Le nombre de feuilles de l’arbre est initialisé par num_leaves  qui vaut  2^8 c’est-à-dire 256.
Learning_rate correspond à la vitesse d’apprentissage qui est de 0.1
Is_unbalance veut dire que le data set est déséquilibré
 (min_split_gain) est de gain minimal qu’il faut afin de couper un arbre
min_child_weight est le poids minimal d’une feuille. Son poids est de 1  afin d’eviter l’overfitting c’est-à-dire le sur-apprentissage. Le sur-apprentissage se produit quand l’algorithme s’accord parfaitement avec les données habituelles avec lesquelles il aura appris mais aura de mauvais résultats si on ajoute de nouvelles données.
Ce phénomène est aussi éviter par le ratui qui est initialisé à 1 par la variable reg_lamba et la partie de données qu’on peut enlever initialisé à 1 par la variable subsample.
'''
    parameters = {
            'num_leaves': 2**8,
            'learning_rate': 0.1,
            'is_unbalance': True,
            'min_split_gain': 0.1,
            'min_child_weight': 1,
            'reg_lambda': 1,
            'subsample': 1,
            'objective':'binary',
            #'device': 'gpu', #comment if you're not using GPU
            'task': 'train'
            }
    #Le nombre de tours est initialisé par la variable  num_rounds qui vaut 300
    num_rounds = 300
    
    # Training
    clf = lgb.train(parameters, lgb_train, num_boost_round=num_rounds)
    
''' 
Nous voulons afficher la matrice de confusion :
On initialise le y_prob aux données de prédiction de X_test, y_pred
La première matrice est initialisée à partir des données de y_test et y_pred
La seconde matrice est initialisée à partir de y_test et y_prob
On met à jour la première matrice en la comparant avec la seconde
Puis on affiche la matrice avec la bibliothèque json
'''
    y_prob = clf.predict(X_test)
    y_pred = fu.binarize_prediction(y_prob, threshold=0.5)
    metrics = fu.classification_metrics_binary(y_test, y_pred)
    metrics2 = fu.classification_metrics_binary_prob(y_test, y_prob)
    metrics.update(metrics2)
    cm = metrics['Confusion Matrix']
    metrics.pop('Confusion Matrix', None)
    
    print(json.dumps(metrics, indent=4, sort_keys=True))
    fu.plot_confusion_matrix(cm, ['no fraud (negative class)', 'fraud (positive class)'])
    
    # On fait une sauvegarde du modèle qu’on a initialisé dans le fichier du projet save. 
    clf.save_model(fu.BASELINE_MODEL)

train_model_lgb()

''' Serveur temps-réel '''
### IMPORTANT ###
# Démarrer le serveur grâce à la commande
# python ./api.py

