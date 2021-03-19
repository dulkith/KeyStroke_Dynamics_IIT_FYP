import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Nearest Neighbor

# nearest neighbor eer, in order
# original, localhost, 10, 25, 50, 100, 250, 500 ms latency
nearest_neighbor_eer_means = [0.14223189891896415, 0.14220462685083024, 0.14234111105492853, 0.1412534786901865, 0.14175821588649423, 0.19606459699338358, 0.2544588088727013, 0.2867328022501085]
nearest_neighbor_eer_std = [0.07834143629103311, 0.07828208617722265, 0.07814054878031151, 0.07776330590689128, 0.07762743612090248, 0.059377802949834094, 0.07368652817334541, 0.08015693062335721]

# nearest neighbor zmfa, in order0.07368652817334541
# original, localhost, 10, 25, 50, 100, 250, 500 ms latency
nearest_neighbor_zmfa_means = [0.7932352941176471, 0.7995098039215688, 0.8, 0.78, 0.783529411764706, 0.9884313725490196, 1.0, 1.0]
nearest_neighbor_zmfa_std = [0.2901163112416172, 0.29091683149150993, 0.29031423137899465, 0.2954375286474847, 0.3092563726698431, 0.06496975311392783, 0.0, 0.0]


plt.suptitle("Nearest Neighbor (mean +/- std)", fontsize=14)
plt.subplot(211)
plt.title("\nEqual Error Rate")
plt.errorbar(np.arange(8), nearest_neighbor_eer_means, nearest_neighbor_eer_std, fmt='ok', lw=3)
plt.yticks(np.arange(0, 0.5, 0.1))
plt.xticks(np.arange(8), labels=["original", "0", "10", "25", "50", "100", "250", "500"])
plt.xlabel("latency (ms)")

plt.subplot(212)
plt.title("\nZero Miss False Alarm Rate")
plt.errorbar(np.arange(8), nearest_neighbor_zmfa_means, nearest_neighbor_zmfa_std, fmt='ok', lw=3)
plt.yticks(np.arange(0.4, 1.1, 0.1))
plt.xticks(np.arange(8), labels=["original", "0", "10", "25", "50", "100", "250", "500"])
plt.xlabel("latency (ms)")

plt.tight_layout(pad=1)
plt.savefig("nearest_neighbor_latency_evolution.png")
plt.show()


# Neural Network

# neural network eer, in order
# original, localhost, 10, 25, 50, 100, 250, 500 ms latency
neural_network_eer_means = [0.17008425741362262, 0.17272773242916364, 0.16951768057182362, 0.17297980948648806, 0.17246014401407966, 0.1746674325641522, 0.18727332431701174, 0.23549438643728693]
neural_network_eer_std = [0.11733277723280358, 0.12033119852472433, 0.11679866667356752, 0.11785177409186871, 0.12209918178046537, 0.12369255108501127, 0.12424592123891964, 0.12601329966944322]

# neural network zmfa, in order
# original, localhost, 10, 25, 50, 100, 250, 500 ms latency
neural_network_zmfa_means = [0.7529411764705882, 0.7562745098039215, 0.7481372549019608, 0.7553921568627451, 0.7434313725490195, 0.8359803921568628, 0.852843137254902, 0.8122549019607842]
neural_network_zmfa_std = [0.31405347438786047, 0.30829042065542184, 0.3179591180779929, 0.3125760192043249, 0.32289057546117, 0.23008101643841983, 0.20683481829891667, 0.19893666773646435]


plt.suptitle("Neural Network (mean +/- std)", fontsize=14)
plt.subplot(211)
plt.title("\nEqual Error Rate")
plt.errorbar(np.arange(8), neural_network_eer_means, neural_network_eer_std, fmt='ok', lw=3)
plt.yticks(np.arange(0, 0.5, 0.1))
plt.xticks(np.arange(8), labels=["original", "0", "10", "25", "50", "100", "250", "500"])
plt.xlabel("latency (ms)")

plt.subplot(212)
plt.title("\nZero Miss False Alarm Rate")
plt.errorbar(np.arange(8), neural_network_zmfa_means, neural_network_zmfa_std, fmt='ok', lw=3)
plt.yticks(np.arange(0.4, 1.1, 0.1))
plt.xticks(np.arange(8), labels=["original", "0", "10", "25", "50", "100", "250", "500"])
plt.xlabel("latency (ms)")

plt.tight_layout(pad=1)
plt.savefig("neural_network_latency_evolution.png")
plt.show()
