import os
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
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from Subject import Subject
from DetectorNearestNeighbor import DetectorNearestNeighbor
from DetectorNeuralNetwork import DetectorNeuralNetwork


def load_dataset(name):
    data = pd.read_csv(name)

    if len(data.columns) > 12:
        # Original dataset
        data = data[data.columns.drop(list(data.filter(regex='H')))]
        data = data[data.columns.drop(list(data.filter(regex='UD')))]

        data_columns = data[data.columns.drop(['sessionIndex', 'rep', 'subject'])].columns
        data = data[data.columns.drop(['sessionIndex', 'rep'])]
        data["subject"] = data["subject"].apply(lambda x: int(x[1:]))

    else:
        # Network dataset
        data = data[data.columns.drop(['Unnamed: 0'])]

    return data


# Original dataset
data = load_dataset("datasets/killourhy_maxion_password.csv")

# Network dataset
# data = load_dataset("network_gns3/data_over_network_0_latency_0_jitter.csv")

# data = load_dataset("network_gns3/data_over_network_10_latency_0_jitter.csv")
# data = load_dataset("network_gns3/data_over_network_25_latency_0_jitter.csv")
# data = load_dataset("network_gns3/data_over_network_50_latency_0_jitter.csv")
# data = load_dataset("network_gns3/data_over_network_100_latency_0_jitter.csv")
# data = load_dataset("network_gns3/data_over_network_250_latency_0_jitter.csv")
# data = load_dataset("network_gns3/data_over_network_500_latency_0_jitter.csv")

# jitter
# data = load_dataset("network_gns3/data_over_network_10_latency_5_jitter.csv")
# data = load_dataset("network_gns3/data_over_network_25_latency_12_jitter.csv")
# data = load_dataset("network_gns3/data_over_network_50_latency_25_jitter.csv")
# data = load_dataset("network_gns3/data_over_network_100_latency_50_jitter.csv")
# data = load_dataset("network_gns3/data_over_network_250_latency_125_jitter.csv")
# data = load_dataset("network_gns3/data_over_network_500_latency_250_jitter.csv")


array_eer = []
array_zm = []
array_1p = []

# Subject management

# Nearest Neighbor

typing_subjects = []
subjects = data.subject.unique().tolist()
nb_subjects = len(data.subject.unique())

for i in subjects:
    typing_subjects.append(Subject(i, data, DetectorNearestNeighbor()))

for subject in typing_subjects:
    subject.train()
    subject.test()
    subject.evaluation()

    array_eer.append(subject.eer)
    array_zm.append(subject.zero_miss_false_alarm)

print("Detector: Nearest Neighbor (Mahalanobis)")
print("Equal Error Rate average:                    " + str(np.mean(array_eer)))
print("Equal Error Rate standard deviation:         (" + str(np.std(array_eer)) + ")")
print("Zero Miss False Alarm average:               " + str(np.mean(array_zm)))
print("Zero Miss False Alarm standard deviation:    (" + str(np.std(array_zm)) + ")")

# Neural Network

typing_subjects = []
subjects = data.subject.unique().tolist()
nb_subjects = len(data.subject.unique())

for i in subjects:
    typing_subjects.append(Subject(i, data, DetectorNeuralNetwork()))

for subject in typing_subjects:
    subject.train()
    subject.test()
    subject.evaluation()

    array_eer.append(subject.eer)
    array_zm.append(subject.zero_miss_false_alarm)

print("Detector: Nearest Network Auto-Associative / Auto-Encoder")
print("Equal Error Rate average:                    " + str(np.mean(array_eer)))
print("Equal Error Rate standard deviation:         (" + str(np.std(array_eer)) + ")")
print("Zero Miss False Alarm average:               " + str(np.mean(array_zm)))
print("Zero Miss False Alarm standard deviation:    (" + str(np.std(array_zm)) + ")")
