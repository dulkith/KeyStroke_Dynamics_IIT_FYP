import pandas as pd
import numpy as np
import sklearn as sc
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.model_selection import train_test_split
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.covariance import EmpiricalCovariance
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from DetectorNearestNeighbor import DetectorNearestNeighbor
from DetectorNeuralNetwork import DetectorNeuralNetwork


class Subject:
    def __init__(self, id, data, detector):
        self.id = id
        self.data = data
        self.detector = detector

        self.impostors = self.data.subject.unique().tolist()
        self.impostors.remove(id)

        self.user_mean = None
        self.user_std = None
        self.user_min = None
        self.user_max = None

        self.impostor_mean = None
        self.impostor_std = None
        self.impostor_min = None
        self.impostor_max = None

        self.global_max = None
        self.global_min = None

        self.fpr = [1]  # User mistaken as impostor
        self.tpr = [1]  # Hit Rate, impostors are detected ( 1 - Miss rate, impostors are missed )
        self.threshold = [0]

        self.y_score = None
        self.y_true = None

        self.eer = None
        self.thresh = None
        self.zero_miss_false_alarm = None

    def train(self):
        s = self.data[self.data["subject"] == self.id]
        X = s[s.columns.drop(['subject'])]

        self.detector.x_train = X.head(200)
        self.detector.x_test = X.tail(200)

        # self.detector.cov = np.cov(self.detector.x_train, rowvar=False)
        # self.detector.cov = EmpiricalCovariance().fit(self.detector.x_train).covariance_
        self.detector.cov = pd.DataFrame.cov(self.detector.x_train)
        self.detector.create_detector()
        self.detector.fit()

    def test(self):
        self.test_user_score()
        self.test_impostor_score()

    def test_user_score(self):
        self.detector.user_score = self.detector.distance_user()

    def test_impostor_score(self):
        i = 0
        for x in self.impostors:
            si = self.data[self.data["subject"] == x]
            Xi = si[si.columns.drop(['subject'])]
            self.detector.xi_test = Xi.head(5)

            tmp_impostor_score = self.detector.distance_impostor()

            if i == 0:
                self.detector.impostor_score = tmp_impostor_score
            else:
                self.detector.impostor_score = np.append(self.detector.impostor_score, tmp_impostor_score)

            i = i + 1

    def evaluation(self):
        self.stats()
        self.metrics_roc()
        self.equal_error_rate()
        self.zero_miss()

    def stats(self):

        self.user_mean = np.mean(self.detector.user_score)
        self.user_std = np.std(self.detector.user_score)
        self.user_min = np.min(self.detector.user_score)
        self.user_max = np.max(self.detector.user_score)

        self.impostor_mean = np.mean(self.detector.impostor_score)
        self.impostor_std = np.std(self.detector.impostor_score)
        self.impostor_min = np.min(self.detector.impostor_score)
        self.impostor_max = np.max(self.detector.impostor_score)

        self.global_max = None
        self.global_min = None

        if self.user_max >= self.impostor_max:
            self.global_max = self.user_max
        else:
            self.global_max = self.impostor_max

        if self.user_min <= self.impostor_min:
            self.global_min = self.user_min
        else:
            self.global_min = self.impostor_min

    def metrics_roc(self):
        # For each possible threshold
        for t in np.arange(0, int(self.global_max), self.detector.step):

            self.y_score = []
            self.y_true = []

            # False alarm rate so False Positive Rate
            # Every user mistaken as impostor
            for e in self.detector.user_score:

                # true label
                self.y_true.append(0)

                # score label
                if e > t:
                    self.y_score.append(1)
                else:
                    self.y_score.append(0)

            # Hit Rate so True Positive Rate
            # Every impostor as user
            for e in self.detector.impostor_score:

                # true label
                self.y_true.append(1)

                # score label
                if e > t:
                    self.y_score.append(1)
                else:
                    self.y_score.append(0)

            # temp_fpr, temp_tpr, temp_threshold = metrics.roc_curve(self.y_true, self.y_score)
            #
            # if temp_fpr[1] != 1:
            #     self.fpr.append(temp_fpr[1])
            #     self.tpr.append(temp_tpr[1])
            #
            #     self.threshold.append(t)

            tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_score).ravel()
            p = tp + fn
            n = fp + tn
            temp_fpr = fp / n
            temp_tpr = tp / p

            self.fpr.append(temp_fpr)
            self.tpr.append(temp_tpr)
            self.threshold.append(t)

        self.fpr.append(0)
        self.tpr.append(0)
        self.threshold.append(int(self.global_max))

    def plot_roc(self):
        plt.figure(figsize=(10, 10))
        plt.plot(self.fpr, self.tpr, color='xkcd:green')
        plt.xlabel('False Positive Rate / False Alarm Rate')
        plt.ylabel('True Positive Rate / Hit Rate ')
        plt.title('ROC cruve')
        plt.show()

    def equal_error_rate(self):
        self.eer = brentq(lambda x: 1. - x - interp1d(self.fpr, self.tpr)(x), 0., 1.)
        self.thresh = interp1d(self.fpr, self.threshold)(self.eer)

    def zero_miss(self):
        self.zero_miss_false_alarm = 1
        for index, val in enumerate(self.tpr):
            if val == 1 and self.fpr[index] < self.zero_miss_false_alarm:
                self.zero_miss_false_alarm = self.fpr[index]