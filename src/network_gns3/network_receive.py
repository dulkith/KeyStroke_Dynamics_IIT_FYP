import time
import pandas as pd
import socket

PORT = 7002


def add_row(columns, row):
    dic = {}
    for index, val in enumerate(row):
        dic[columns[index]] = val
    return dic


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
    sock.bind(('', PORT))
    sock.listen(1)
    newsock, (client_host, client_port) = sock.accept()

    rounds = 0
    max_rounds = 400
    max_features = 11
    columns = ['subject', 'DD.period.t', 'DD.t.i', 'DD.i.e', 'DD.e.five', 'DD.five.Shift.r', 'DD.Shift.r.o', 'DD.o.a',
               'DD.a.n', 'DD.n.l', 'DD.l.Return']
    subjects = [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]

    df = pd.DataFrame(columns=columns)
    rows_list = []
    row = []

    message = newsock.recv(1)
    old = time.perf_counter()

    for user in subjects:
        for i in range(max_rounds):
            row = []
            print("Receiving Subject: " + str(user) + " ---------- Iteration: " + str(i) + " ---------- ")
            row.append(user)

            for j in range(len(columns)-1):
                message = newsock.recv(1)
                chrono = time.perf_counter()
                timing = chrono - old

                row.append(timing)
                # print(timing)
                # print(message.decode())

                old = chrono

            # print(row)
            rows_list.append(add_row(columns, row))
            # print("\n")

    df = pd.DataFrame(rows_list, columns=columns)
    print(df)

    df.to_csv('data_over_network_X.csv')

    newsock.close()
