# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:29:52 2019

@author: Lambert Rosique
@source : https://github.com/miguelgfierro/sciblog_support/tree/master/Intro_to_Fraud_Detection
"""
from flask import (Flask, request, abort, jsonify, make_response,
                   render_template)
from flask_socketio import SocketIO, emit
import pandas as pd
import lightgbm as lgb
import fraud_utils as fu
import os
import time

"""Initialisation les constantes BAD_REQUEST à 400,
 STATUS_OK à 200 ; NOT_FOUND à 404, SERVER_ERROR à 500."""
 
BAD_REQUEST = 400
STATUS_OK = 200
NOT_FOUND = 404
SERVER_ERROR = 500

"""initialise notre application grâce à la bibliothèque Flask."""

app = Flask(__name__)  # app
app.static_folder = 'static'  # define static folder for css, img, js
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

"""Recherche d'erreur en utilisant la fonction errorhandler et en lui 
passant en paramètres les constantes BAD_REQUEST, NOT_FOUND, SERVER_ERROR ."""

@app.errorhandler(BAD_REQUEST)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad request'}), BAD_REQUEST)


@app.errorhandler(NOT_FOUND)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), NOT_FOUND)


@app.errorhandler(SERVER_ERROR)
def server_error(error):
    return make_response(jsonify({'error': 'Server error'}), SERVER_ERROR)

#On met l’application en route.
@app.route('/')

#le serveur fonctionne correctement
def hello():
    return "Le serveur fonctionne correctement"


@app.route('/map')
#retourne l’adresse du fichier dans lequel la carte est crée
def map():
    return render_template('index.html')


"""Test que quand l’application est connectée le client 
l’est également sinon il est déconnecté dans les fonctions de test. """

@socketio.on('connect', namespace='/fraud')
def test_connect():
    print('Client connected')


@socketio.on('disconnect', namespace='/fraud')
def test_disconnect():
    print('Client disconnected')

#emet la variable my_pong au serveur
@socketio.on('my_ping', namespace='/fraud')
def ping_pong():
   
    emit('my_pong')

#retourne si l’url est accessible
@app.route('/health')
def health_check():

    socketio.emit('health_signal',
                  {'data': 'HEALTH CHECK', 'note': 'OK'},
                  broadcast=True,
                  namespace='/fraud')
    return make_response(jsonify({'health': 'OK'}), STATUS_OK)


def manage_query(request):
    if not request.is_json:
        abort(BAD_REQUEST)
    dict_query = request.get_json()
    X = pd.DataFrame(dict_query, index=[0])
    return X


@app.route('/predict', methods=['POST'])
def predict():
   
    X = manage_query(request)
    y_pred = model.predict(X)[0]
    return make_response(jsonify({'fraud': y_pred}), STATUS_OK)



#retourne  du y_pred est initialisée grâce au fichier fraud utils 
#soketio met à jour la carte quand il ya une nouvelle fraude qui
#est détéctée 

@app.route('/predict_map', methods=['POST'])
def predict_map():
   
    X = manage_query(request)
    y_pred = model.predict(X)[0]
    print("Value predicted: {}".format(y_pred))
    if y_pred >= fu.FRAUD_THRESHOLD:
        row = fu.select_random_row_cities()
        location = {"country":row["country"].iloc[0], "title": row["city"].iloc[0], "latitude": row["latitude"].iloc[0], "longitude": row["longitude"].iloc[0]}
        print("New location: {}".format(location))
        socketio.emit('map_update', location, broadcast=True, namespace='/fraud')
    return make_response(jsonify({'fraud': y_pred}), STATUS_OK)


"""On initialise le nombre d’essais maximum à 120.
Tant qu’on ne dépasse pas ce nombre d’essaie et que le serveur n’est pas en ligne la fonction renvoie que le serveur n’est pas prêt
Si le compteur est égal au nombre max d’essais on renvoie que le maximum d’essais a été atteint.
Sinon le serveur est prêt et on peut regarder notre résultat sur l’URL indiquée
L’algo détermine et stocke les valeurs frauduleuses puis affiche les fraudes sur la carte grâce à json. 
La carte est sur l’url initialisé. On utilise un compteur tant que i ne dépasse pas le nombre de point qu’on veut.
"""

max_try=120
cpt = 0
while cpt < max_try and not os.path.exists(fu.BASELINE_MODEL):
    cpt += 1
    print("No model found ("+str(cpt)+") : "+fu.BASELINE_MODEL)
    time.sleep(1)
if cpt == max_try:
    raise Exception("Launch aborted")
model = lgb.Booster(model_file=fu.BASELINE_MODEL)

"""Dans le main on met en route le serveur et application et à la fin on les arrête. """
if __name__ == "__main__":
    try:
        print("Server started")
        socketio.run(app, debug=True)
    except:
        raise
    finally:
        print("Stop procedure")
