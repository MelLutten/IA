# Light GBM algorithm #

## Installation of your environnement ##

### Install Python ###
Go to the website of [Python](https://www.python.org/downloads/) and download the last version.

### Install Anaconda

Visit the Anaconda homepage. Then click [Anaconda](https://www.anaconda.com/) from the menu and click [Download](https://www.anaconda.com/distribution/) to go to the download page. Once it's downloaded you follow the installation wizard.
The installation should take less than 10 minutes and take up a little more than 1 GB of space on your hard drive.

### Install libraries  ###

```bash
pip install numpy
pip install matplotlib
pip install pandas
pip install lightgbm
pip install scikit-learn
pip install requests
pip install flask
pip install flask_socketio
```
* *numpy*, *matplotlib* and *pandas* are standard libraries of learning machine. They allow users to respectively working with vectors, draw graphs and manipulate data. 
* *lightgbm* is a library. It contains implementation of the LightGBM algorithm and *scikit-learn* is a more general library of learning machine with more tools. 
* *requests*, *flask* et *flask_socketio* allow to create and to call a server.s

## Launch application ##

### Launch server ###

To start the server you need to launch the command below : 

```
python ./api.py
```

You should have on your terminal : 
 ``` Bash 
 
 Server started
    * Restarting with stat
 Server started 
    * Debugger is active !
    * Debugger PIN : 171-944-202

```

### Launch file : real-time-fraud-detection-map.py ###

You compile on Anaconda (spider) the real-time-fraud-detection-map.py file and then on the output a link appears. When you click on it, you are redirected on the given address. You are now on the map with different fraud. 

### Unzip ###

If you want the code to work you need to unzip the data repository to access to csv files that contain data about credit cards and cities. 
