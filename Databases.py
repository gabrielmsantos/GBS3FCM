import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from Helpers import extract_features_and_labels


def get_bupa(shuffle_data=True):
    filepath_bupa = './data/bupa/bupa.data'
    X, Y_hat = extract_features_and_labels(filepath_bupa, shuffle_data)
    Y_hat = Y_hat - 1
    return X, Y_hat


def get_dermatology(shuffle_data=True):
    filepath_dermatology = './data/dermatology/dermatology.data'
    X, Y_hat = extract_features_and_labels(filepath_dermatology, shuffle_data)
    X = X[:, :-1]  # Remove the last column (age)
    Y_hat = Y_hat - 1

    return X, Y_hat


def get_diabetes(shuffle_data=True):
    filepath_diabetes = './data/diabetes/diabetes.csv'
    # Read the CSV file into a DataFrame
    data = pd.read_csv(filepath_diabetes)

    # Shuffle the data
    if shuffle_data:
        data = shuffle(data)

    # Separate the features and the labels
    X = data.iloc[:, :-1].values  # All columns except the last
    Y_hat = data.iloc[:, -1].values  # Only the last column

    return X, Y_hat


def get_heart(shuffle_data=True):
    filepath_heart = './data/heart/processed.cleveland.data'
    X, Y_hat = extract_features_and_labels(filepath_heart, shuffle_data)
    Y_hat = Y_hat - 1
    return X, Y_hat


def get_waveform(shuffle_data=True):
    filepath_waveform = './data/waveform/waveform.data'
    X, Y_hat = extract_features_and_labels(filepath_waveform, shuffle_data)
    return X, Y_hat


def get_wdbc(shuffle_data=True):
    # Read the CSV file into a DataFrame
    data = pd.read_csv('./data/wdbc/wdbc.data')

    # Shuffle the data
    if shuffle_data:
        data = shuffle(data)

    # Assuming the first column is ID, the second column is diagnosis, and the rest are features
    features = data.iloc[:, 2:].values  # All columns except the first two (ID and diagnosis)
    diagnosis = data.iloc[:, 1].values  # Only the second column (diagnosis)

    # Transform diagnosis values: 'M' -> 0, 'B' -> 1
    diagnosis = np.where(diagnosis == 'M', 0, 1)

    return features, diagnosis


def get_gauss50(n_samples, n_features=50, shuffle_data=True):
    mean_class1 = np.full(n_features, 0.23)
    mean_class2 = np.full(n_features, -0.23)
    cov = np.eye(n_features)

    X_class1 = np.random.multivariate_normal(mean_class1, cov, n_samples // 2)
    X_class2 = np.random.multivariate_normal(mean_class2, cov, n_samples // 2)

    X = np.vstack((X_class1, X_class2))
    Y = np.hstack((np.ones(n_samples // 2), 0 * np.ones(n_samples // 2))).astype(int)

    if shuffle_data:
        X, Y = shuffle(X, Y)

    return X, Y


def get_gauss50x(n_samples, n_features=50, shuffle_data=True):
    mu1 = np.full(n_features, 0.25)
    mu2 = np.full(n_features, -0.25)
    cov = np.eye(n_features)

    def generate_mixture_samples(mean1, mean2, cov, size):
        samples1 = np.random.multivariate_normal(mean1, cov, int(size * 0.49))
        samples2 = np.random.multivariate_normal(mean2, cov, int(size * 0.51))
        return np.vstack((samples1, samples2))

    X_class1 = generate_mixture_samples(mu1, mu2, cov, n_samples // 2)
    X_class2 = generate_mixture_samples(-mu1, -mu2, cov, n_samples // 2)

    X = np.vstack((X_class1, X_class2))
    Y = np.hstack((np.ones(n_samples // 2), 0 * np.ones(n_samples // 2))).astype(int)

    if shuffle_data:
        X, Y = shuffle(X, Y)

    return X, Y


