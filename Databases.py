import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from Helpers import extract_features_and_labels


def get_bupa(shuffle_data=True):
    filepath_bupa = './data/bupa/bupa.data'
    X, Y_hat = extract_features_and_labels(filepath_bupa, shuffle_data)

    # Transform the last column (DRINKS) into binary labels
    Y_hat = np.where(X[:, -1] < 3, 0, 1)# The last column (DRINKS)

    # Remove the last column (DRINKS) and the first column (SELECTOR
    X = X[:, :-1]  # Remove the last column (SELECTOR)

    #Y_hat = Y_hat - 1
    return X, Y_hat, 'BUPA'


def get_dermatology(shuffle_data=True):
    filepath_dermatology = './data/dermatology/dermatology.data'
    X, Y_hat = extract_features_and_labels(filepath_dermatology, shuffle_data)
    X = X[:, :-1]  # Remove the last column (age)
    Y_hat = Y_hat - 1

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    return X, Y_hat, 'DERMATOLOGY'


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

    # Standardize the data
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    return X, Y_hat, 'DIABETES'


def get_heart(shuffle_data=True):
    filepath_heart = './data/heart/processed.cleveland.data'
    X, Y_hat = extract_features_and_labels(filepath_heart, shuffle_data)
    Y_hat = np.where(Y_hat > 1, 1, Y_hat)

    # Remove the third column
    # X = np.delete(X, 4, axis=1)

    # Standardize the data
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    return X, Y_hat, 'HEART'


def get_waveform(shuffle_data=True):
    filepath_waveform = './data/waveform/waveform.data'
    X, Y_hat = extract_features_and_labels(filepath_waveform, shuffle_data)

    total = 1000
    X = X[:total]
    Y_hat = Y_hat[:total]

    # Standardize the data
    #mean = np.mean(X, axis=0)
    #std = np.std(X, axis=0)
    #X = (X - mean) / std

    return X, Y_hat, 'WAVEFORM'


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

    # Standardize the data
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = (features - mean) / std

    return features, diagnosis, 'WDBC'


def get_gauss50(n_samples=1550, n_features=50, shuffle_data=True):
    mean_class1 = np.full(n_features, 0.23)
    mean_class2 = np.full(n_features, -0.23)
    cov = np.eye(n_features)

    X_class1 = np.random.multivariate_normal(mean_class1, cov, n_samples // 2)
    X_class2 = np.random.multivariate_normal(mean_class2, cov, n_samples // 2)

    X = np.vstack((X_class1, X_class2))
    Y = np.hstack((np.ones(n_samples // 2), 0 * np.ones(n_samples // 2))).astype(int)

    if shuffle_data:
        X, Y = shuffle(X, Y)

    # Standardize the data

    return X, Y, 'GAUSS50'


def get_gauss50x(n_samples=500, n_features=50, shuffle_data=True):
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
    Y = np.hstack((np.ones(X_class1.shape[0]), 0 * np.ones(X_class2.shape[0]))).astype(int)

    if shuffle_data:
        X, Y = shuffle(X, Y)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    return X, Y, 'GAUSS50x'


def select_database(db_name):
    switcher = {
        'BUPA': get_bupa,
        'DERMATOLOGY': get_dermatology,
        'DIABETES': get_diabetes,
        'HEART': get_heart,
        'WAVEFORM': get_waveform,
        'WDBC': get_wdbc,
        'GAUSS50': get_gauss50,
        'GAUSS50x': get_gauss50x
    }
    # Get the function from switcher dictionary
    func = switcher.get(db_name, lambda: "Invalid database name")
    # Execute the function
    return func()