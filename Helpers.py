import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def extract_features_and_labels(file_path, shuffle_data=True):
    # Read the data file
    data_raw = pd.read_csv(file_path, header=None)

    # Remove rows with missing values
    data = data_raw.replace('?', pd.NA).dropna()

    # Shuffle the data
    if shuffle_data:
        data = shuffle(data)

    # Convert all columns to numeric, non-convertible values will be replaced with NaN
    data = data.apply(pd.to_numeric, errors='coerce')

    # Separate the features and the labels
    X = data.iloc[:, :-1].values  # All columns except the last
    Y = data.iloc[:, -1].values  # Only the last column

    return X, Y


def split_data(X, Y, P, W):
    # Calculate the number of labeled instances
    num_labeled = int(len(X) * P / 100)

    # Split X into labeled and unlabeled instances
    X_l = X[:num_labeled]
    X_u = X[num_labeled:]

    # Split Y into labels for labeled and unlabeled instances
    Y_l = Y[:num_labeled].copy()
    # Y_u = Y[num_labeled:]

    # Calculate the number of mislabeled instances
    num_mislabeled = int(len(Y_l) * W / 100)

    # Randomly select instances to mislabel
    mislabeled_indices = np.random.choice(num_labeled, num_mislabeled, replace=False)

    # Get the unique labels
    unique_labels = np.unique(Y)

    # Mislabel the selected instances
    for index in mislabeled_indices:
        # Choose a new label that is different from the current label
        new_label = np.random.choice(unique_labels)
        while new_label == Y_l[index]:
            new_label = np.random.choice(unique_labels)

        # Assign the new label
        Y_l[index] = new_label

    return X_l, X_u, Y_l