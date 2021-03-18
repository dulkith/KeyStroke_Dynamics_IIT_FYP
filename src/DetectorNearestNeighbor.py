from sklearn.neighbors import NearestNeighbors
import numpy as np


class DetectorNearestNeighbor:
    def __init__(self):
        self.nearest_neighbor = None

        self.x_train = None
        self.x_test = None
        self.xi_test = None
        self.cov = None

        self.user_score = None
        self.impostor_score = None

        self.step = 1

    def create_detector(self):
        self.nearest_neighbor = NearestNeighbors(n_neighbors=1, metric='mahalanobis', metric_params={'V': self.cov, 'VI': np.linalg.inv(self.cov)},
                         algorithm='brute')

    def fit(self):
        self.nearest_neighbor.fit(self.x_train)

    def distance_user(self):
        score, index = self.nearest_neighbor.kneighbors(self.x_test)
        # print(score)
        return score

    def distance_impostor(self):
        score, index = self.nearest_neighbor.kneighbors(self.xi_test)
        # print(score)
        return score

