import os
import pandas as pd
import time
import socket

SERVER = '127.0.0.1'
PORT = 7002


def select_char(name):
    characters = name.split(".")

    char = ''
    if characters[1] == "period":
        char = '.'
    elif characters[1] == "five":
        char = '5'
    elif characters[1] == "Shift":
        char = 'R'
    else:
        char = characters[1]

    return char


# Data gathering
data = pd.read_csv("dataset.csv")

# data = data[data.columns.drop(list(data.filter(regex='DD')))]
data = data[data.columns.drop(list(data.filter(regex='H')))]
data = data[data.columns.drop(list(data.filter(regex='UD')))]

data_columns = data[data.columns.drop(['sessionIndex', 'rep', 'subject'])].columns
data = data[data.columns.drop(['sessionIndex', 'rep'])]
data["subject"] = data["subject"].apply(lambda x: int(x[1:]))
subjects = data.subject.unique().tolist()

# Create TCP connection
ctr = 0

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((SERVER, PORT))

    for user in subjects:

        X = data[data["subject"] == user]
        X = X[X.columns.drop(['subject'])]

        for i, row in X.iterrows():
            print("Sending Subject: " + str(user) + " ---------- Iteration: " + str(i%400) + " ---------- ")

            for j, column in row.iteritems():
                wait_time = column
                ctr = ctr + 1

                char = select_char(j)

                # print("Sending: " + str(char))
                sock.send(char.encode())

                # print("Sleeping:" + str(wait_time))
                time.sleep(wait_time)

            # print("\n")