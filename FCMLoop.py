import numpy as np

E = 10e-10


def initialize_U_FCM(X, V, m=2):
    n, c = X.shape[0], V.shape[0]  # n data points, c clusters
    U = np.zeros((c, n))  # Initialize the U matrix
    for k in range(n):  # Loop over clusters
        for i in range(c):  # Loop over data points
            internal_sum = 0
            d_ik = np.linalg.norm(X[k] - V[i])  # Euclidean distance from X[k] to V[i]
            if d_ik < E:
                d_ik = E
            for j in range(c):
                d_jk = np.linalg.norm(X[k] - V[j])  # Euclidean distance from X[k] to V[j]
                if d_jk < E:
                    d_jk = E
                internal_sum += np.nan_to_num((d_ik / d_jk) ** (2 / (m - 1)))
            U[i, k] = np.nan_to_num(1 / internal_sum)
    return U


def initialize_ULabeled(n_clusters, Y_l):
    U_l = np.zeros((n_clusters, len(Y_l)))  # Initialize the U matrix
    for k in range(len(Y_l)):
        label = Y_l[k]  # Label of instance k
        U_l[label, k] = 1  # Set u_ik = 1
    return U_l


def update_U_labeled(X_l, V, W, U_prev_un, s, lambda1, lambda2, F):
    n, c = X_l.shape[0], V.shape[0]  # n data points, c clusters
    U_l = np.zeros((c, n))  # initialize the U matrix

    # compute matrix P (n  x c) and Q (n x c)
    P = np.zeros((c, n))
    Q = np.zeros((c, n))

    for k in range(n):  # Loop over data points
        for i in range(c):  # Loop over clusters
            d_ik = np.linalg.norm(X_l[k] - V[i])  # Euclidean distance from X[k] to V[i]
            if d_ik < E:
                d_ik = E
            sum_Wkr_ur = np.dot(W[k, :], U_prev_un[i, :])  # Sum over all r, skipping the current i
            P[i, k] = lambda1 * s[k] * F[i, k] * d_ik ** 2 + lambda2 * (2 / (s[k] + 1) - 1) * sum_Wkr_ur

            Q[i, k] = d_ik ** 2 + lambda1 * s[k] * d_ik ** 2 + lambda2 * (2 / (s[k] + 1) - 1) * np.sum(W[k, :])

    for k in range(n):  # Loop over data points
        for i in range(c):  # Loop over clusters
            U_l[i, k] = (P[i, k] + (1 - np.sum(P[:, k] / Q[:, k])) / np.sum(1 / Q[:, k])) / Q[i, k]

    return U_l