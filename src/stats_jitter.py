import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Nearest Neighbor

# nearest neighbor eer, in order
# original, localhost, 10/5, 25/12, 50/25, 100/50, 500/250 ms latency/jitter
nearest_neighbor_eer_means = [0.14223189891896415, 0.14220462685083024, 0.14099755381499768, 0.14216512215852706, 0.15127493581192603, 0.21970294639697946, 0.27055599517004986, 0.27836927863777994]
nearest_neighbor_eer_std = [0.07834143629103311, 0.07828208617722265, 0.07817591850966392, 0.07993296096310204, 0.07814279246773788, 0.0644931467377202, 0.08468839863874482, 0.08262940169322812]

# nearest neighbor zmfa, in order
# original, localhost, 10/5, 25/12, 50/25, 100/50, 500/250 ms latency/jitter
nearest_neighbor_zmfa_means = [0.7932352941176471, 0.7995098039215688, 0.7750980392156862, 0.792156862745098, 0.8985294117647059, 0.9539215686274509, 1.0, 1.0]
nearest_neighbor_zmfa_std = [0.2901163112416172, 0.29091683149150993, 0.28071074044303485, 0.271562752113127, 0.1955191533610621, 0.1116599956691452, 0.0, 0.0]


plt.suptitle("Nearest Neighbor (mean +/- std)", fontsize=14)
plt.subplot(211)
plt.title("\nEqual Error Rate")
plt.errorbar(np.arange(8), nearest_neighbor_eer_means, nearest_neighbor_eer_std, fmt='ok', lw=3)
plt.yticks(np.arange(0, 0.5, 0.1))
plt.xticks(np.arange(8), labels=["original", "0/0", "10/5", "25/12", "50/25", "100/50", "250/125", "500/250"])
plt.xlabel("latency/jitter (ms)")

plt.subplot(212)
plt.title("\nZero Miss False Alarm Rate")
plt.errorbar(np.arange(8), nearest_neighbor_zmfa_means, nearest_neighbor_zmfa_std, fmt='ok', lw=3)
plt.yticks(np.arange(0.4, 1.1, 0.1))
plt.xticks(np.arange(8), labels=["original", "0/0", "10/5", "25/12", "50/25", "100/50", "250/125", "500/250"])
plt.xlabel("latency/jitter (ms)")

plt.tight_layout(pad=1)
plt.savefig("nearest_neighbor_jitter_evolution.png")
plt.show()


# Neural Network

# neural network eer, in order
# original, localhost, 10/5, 25/12, 50/25, 100/50, 250/125ms, 500/250 ms latency/jitter
neural_network_eer_means = [0.17008425741362262, 0.17272773242916364, 0.1737979964062586, 0.17140370223815893, 0.17095535771856324, 0.177107260600753, 0.2156767746804873, 0.26126432198338834]
neural_network_eer_std = [0.11733277723280358, 0.12033119852472433, 0.1202692698269893, 0.11656724777305576, 0.11779726933507147, 0.1192999089159846, 0.11420899922204342, 0.10976545694180807]

# neural network zmfa, in order
# original, localhost, 10/5, 25/12, 50/25, 100/50, 250/125ms, 500/250 ms latency/jitter
neural_network_zmfa_means = [0.7529411764705882, 0.7562745098039215, 0.7573529411764707, 0.7487254901960781, 0.7536274509803921, 0.8129411764705884, 0.8769607843137257, 0.9556862745098039]
neural_network_zmfa_std = [0.31405347438786047, 0.30829042065542184, 0.29991559750803953, 0.31404514964717506, 0.3079614796885056, 0.24262691538492623, 0.13399253136697564, 0.0810411394428728]


plt.suptitle("Neural Network (mean +/- std)", fontsize=14)
plt.subplot(211)
plt.title("\nEqual Error Rate")
plt.errorbar(np.arange(8), neural_network_eer_means, neural_network_eer_std, fmt='ok', lw=3)
plt.yticks(np.arange(0, 0.5, 0.1))
plt.xticks(np.arange(8), labels=["original", "0/0", "10/5", "25/12", "50/25", "100/50", "250/125", "500/250"])
plt.xlabel("latency/jitter (ms)")

plt.subplot(212)
plt.title("\nZero Miss False Alarm Rate")
plt.errorbar(np.arange(8), neural_network_zmfa_means, neural_network_zmfa_std, fmt='ok', lw=3)
plt.yticks(np.arange(0.4, 1.1, 0.1))
plt.xticks(np.arange(8), labels=["original", "0/0", "10/5", "25/12", "50/25", "100/50", "250/125", "500/250"])
plt.xlabel("latency/jitter (ms)")

plt.tight_layout(pad=1)
plt.savefig("neural_network_jitter_evolution.png")
plt.show()
