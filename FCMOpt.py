import numpy as np
from sklearn.cluster import KMeans

E = 10e-10


def initialize_U_FCM(X, V, m=2):
    n, c = X.shape[0], V.shape[0]  # n data points, c clusters
    U = np.zeros((c, n))  # Initialize the U matrix

    # Compute the distance matrix between each data point and each cluster center
    distances = np.linalg.norm(X[:, np.newaxis, :] - V[np.newaxis, :, :], axis=2)  # shape (n, c)
    distances[distances < E] = E  # Avoid division by zero

    # Compute the membership matrix U
    power = 2 / (m - 1)
    inv_distances = 1 / distances
    inv_distances_power = inv_distances ** power

    for i in range(c):
        U[i, :] = inv_distances_power[:, i] / np.sum(inv_distances_power, axis=1)

    return np.nan_to_num(U)


def initialize_ULabeled(n_clusters, Y_l):
    U_l = np.zeros((n_clusters, len(Y_l)))  # Initialize the U matrix
    U_l[Y_l, np.arange(len(Y_l))] = 1  # Set u_ik = 1 for the correct indices
    return U_l


def initialize_V(X_l, Y_l, labels):
    unique_labels = np.unique(labels)
    # Determine the number of classes
    # n_classes = unique_labels.max() + 1  # Assuming labels are 0-indexed
    n_classes = unique_labels.size
    # Create an empty array of shape (n_classes, n_features)
    centers = np.zeros((n_classes, X_l.shape[1]))
    for label in unique_labels:
        # Compute the mean for each class and assign it to the correct position in the array
        centers[label] = np.nan_to_num(X_l[Y_l == label].mean(axis=0))
    return centers


def initialize_V_KM(X_l, Y_hat):
    # Use k-means++ or other sophisticated methods for initialization
    kmeans = KMeans(n_clusters=len(np.unique(Y_hat)), init='k-means++')
    kmeans.fit(X_l)
    V = kmeans.cluster_centers_
    return V


def initialize_S(size):
    return np.full(size, 0.5)


def update_U_labeled(X_l, V, W, U_prev_un, s, lambda1, lambda2, F):
    #n, c = X_l.shape[0], V.shape[0]  # n data points, c clusters

    # Compute distances
    distances = np.linalg.norm(X_l[:, np.newaxis, :] - V[np.newaxis, :, :], axis=2)  # shape (n, c)
    distances[distances < E] = E  # Avoid division by zero
    distances_squared = distances ** 2  # shape (n, c)

    # Compute sum_Wkr_ur for all data points and clusters
    sum_Wkr_ur = np.dot(W, U_prev_un.T)  # shape (n, c)

    # Prepare s for broadcasting
    s_expanded = s[:, np.newaxis]  # shape (n, 1)
    s_broadcasted = s_expanded.T  # shape (1, n)

    # Compute P and Q matrices
    term1 = lambda1 * s_broadcasted * F  # shape (c, n)
    term2 = lambda2 * (2 / (s_broadcasted + 1) - 1)  # shape (1, n)

    P = term1 * distances_squared.T + term2 * sum_Wkr_ur.T  # shape (c, n)
    Q = distances_squared.T + lambda1 * s_broadcasted * distances_squared.T + term2 * np.sum(W, axis=1).reshape(1, -1)  # shape (c, n)

    # Compute U_l matrix
    U_l = (P + (1 - np.sum(P / Q, axis=0)) / np.sum(1 / Q, axis=0)) / Q  # shape (c, n)

    return U_l