import pandas as pd


def value_above_thresh(row):
    for r in row:
        if r >= 10:
            print(row)
            break


data = pd.read_csv("data_over_network_0_latency_0_jitter.csv")
data = data[data.columns.drop(['Unnamed: 0', 'subject'])]


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
data = load_dataset("dataset.csv")

data = data[data.columns.drop(['subject'])]

print(len(data.columns))
print(data.mean().mean())