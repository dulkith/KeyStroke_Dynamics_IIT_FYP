import pandas as pd
import numpy as np
import sklearn as sc
import matplotlib.pyplot as plt
import scipy.stats
from sklearn import metrics
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.model_selection import train_test_split
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.pyplot import figure
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Dense, Input


class DetectorNeuralNetwork:
    def __init__(self):
        self.neural_network = None

        self.x_train = None
        self.x_test = None
        self.xi_test = None
        self.cov = None
        self.weights = None

        self.user_score = None
        self.impostor_score = None

        self.step = 0.01

    def create_detector(self):
        dim = self.x_train.shape[1]
        input = Input(shape=(dim,))
        hidden = Dense(dim, activation="sigmoid",
                       kernel_initializer='random_normal',
                       bias_initializer='zeros')(input)
        output = Dense(dim, activation="linear",
                       kernel_initializer='random_normal',
                       bias_initializer='zeros')(hidden)

        self.neural_network = Model(inputs=input, outputs=output)

        # autoencoder.summary()

        opt = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.0003)
        self.neural_network.compile(optimizer=opt, loss='mse')

    def fit(self):
        self.neural_network.fit(
            self.x_train, self.x_train, epochs=500, verbose=0)
        self.weights = self.neural_network.get_weights()

    def distance_user(self):
        predictions = self.neural_network.predict(self.x_test)
        user_score = []

        for i in range(0, self.x_test.shape[0]):
            dist = distance.euclidean(self.x_test.iloc[i], predictions[i])
            user_score.append(dist)

        return user_score

    def distance_impostor(self):
        predictions = self.neural_network.predict(self.xi_test)
        impostor_score = []

        for i in range(0, self.xi_test.shape[0]):
            dist = distance.euclidean(self.xi_test.iloc[i], predictions[i])
            impostor_score.append(dist)

        return impostor_score
