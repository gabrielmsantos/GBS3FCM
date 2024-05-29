import numpy as np
import cvxopt
from math import sqrt
from scipy.spatial import distance
from Helpers import split_data, extract_features_and_labels


E = 10e-10  # Small constant to avoid division by zero


def initialize_V(X_l, Y_l, labels):
    """
    Initialize the cluster centers V.

    Args:
    X_l (np.ndarray): Labeled data points.
    Y_l (np.ndarray): Corresponding labels for the data points.
    labels (np.ndarray): Unique labels.

    Returns:
    np.ndarray: Initialized cluster centers.
    """
    unique_labels = np.unique(labels)
    n_classes = unique_labels.size
    centers = np.zeros((n_classes, X_l.shape[1]))
    for label in unique_labels:
        centers[label] = np.nan_to_num(X_l[Y_l == label].mean(axis=0))
    return centers


def initialize_U(X, V, m=2):
    """
    Initialize the membership matrix U.

    Args:
    X (np.ndarray): Data points.
    V (np.ndarray): Cluster centers.
    m (int): Fuzziness parameter.

    Returns:
    np.ndarray: Initialized membership matrix.
    """
    n, c = X.shape[0], V.shape[0]
    U = np.zeros((c, n))
    for k in range(n):
        for i in range(c):
            d_ik = np.linalg.norm(X[k] - V[i])
            d_ik = max(d_ik, E)
            internal_sum = sum((d_ik / max(np.linalg.norm(X[k] - V[j]), E)) ** (2 / (m - 1)) for j in range(c))
            U[i, k] = 1 / internal_sum
    return U


def initialize_S(size):
    """
    Initialize the confidence levels S.

    Args:
    size (int): Number of labeled data points.

    Returns:
    np.ndarray: Initialized confidence levels.
    """
    return np.full(size, 0.5)


def initialize_F(X_l, Y_l, labels):
    """
    Initialize the matrix F for labeled data.

    Args:
    X_l (np.ndarray): Labeled data points.
    Y_l (np.ndarray): Corresponding labels for the data points.
    labels (np.ndarray): Unique labels.

    Returns:
    np.ndarray: Initialized F matrix.
    """
    unique_labels = np.unique(labels)
    c = len(unique_labels)
    n = len(X_l)
    F = np.zeros((c, n))
    label_mapping = {label: index for index, label in enumerate(unique_labels)}
    for k in range(n):
        F[label_mapping[Y_l[k]], k] = 1
    return F


def update_u(X_l, X_un, V, W, U_prev_l, U_prev_un, s, lambda1, lambda2, F):
    """
    Update the membership matrices for labeled and unlabeled data.

    Args:
    X_l (np.ndarray): Labeled data points.
    X_un (np.ndarray): Unlabeled data points.
    V (np.ndarray): Cluster centers.
    W (np.ndarray): Weight matrix.
    U_prev_l (np.ndarray): Previous membership matrix for labeled data.
    U_prev_un (np.ndarray): Previous membership matrix for unlabeled data.
    s (np.ndarray): Confidence levels.
    lambda1 (float): Regularization parameter for labeled data.
    lambda2 (float): Regularization parameter for unlabeled data.
    F (np.ndarray): Matrix F for labeled data.

    Returns:
    tuple: Updated membership matrices for labeled and unlabeled data.
    """
    U_l = update_U_labeled(X_l, V, W, U_prev_un, s, lambda1, lambda2, F)
    U_un = update_U_unlabeled(X_un, V, W, U_prev_l, s, lambda2, F)
    return U_l, U_un


def update_U_labeled(X_l, V, W, U_prev_un, s, lambda1, lambda2, F):
    """
    Update the membership matrix for labeled data.

    Args:
    X_l (np.ndarray): Labeled data points.
    V (np.ndarray): Cluster centers.
    W (np.ndarray): Weight matrix.
    U_prev_un (np.ndarray): Previous membership matrix for unlabeled data.
    s (np.ndarray): Confidence levels.
    lambda1 (float): Regularization parameter for labeled data.
    lambda2 (float): Regularization parameter for unlabeled data.
    F (np.ndarray): Matrix F for labeled data.

    Returns:
    np.ndarray: Updated membership matrix for labeled data.
    """
    n, c = X_l.shape[0], V.shape[0]
    U_l = np.zeros((c, n))
    P = np.zeros((c, n))
    Q = np.zeros((c, n))

    for k in range(n):
        for i in range(c):
            d_ik = np.linalg.norm(X_l[k] - V[i])
            d_ik = max(d_ik, E)
            sum_Wkr_ur = np.dot(W[k, :], U_prev_un[i, :])
            P[i, k] = lambda1 * s[k] * F[i, k] * d_ik ** 2 + lambda2 * (2 / (s[k] + 1) - 1) * sum_Wkr_ur
            Q[i, k] = d_ik ** 2 + lambda1 * s[k] * d_ik ** 2 + lambda2 * (2 / (s[k] + 1) - 1) * np.sum(W[k, :])

    for k in range(n):
        for i in range(c):
            U_l[i, k] = (P[i, k] + (1 - np.sum(P[:, k] / Q[:, k])) / np.sum(1 / Q[:, k])) / Q[i, k]

    return U_l


def update_U_unlabeled(X_u, V, W, U_prev_l, s, lambda2, F):
    """
    Update the membership matrix for unlabeled data.

    Args:
    X_u (np.ndarray): Unlabeled data points.
    V (np.ndarray): Cluster centers.
    W (np.ndarray): Weight matrix.
    U_prev_l (np.ndarray): Previous membership matrix for labeled data.
    s (np.ndarray): Confidence levels.
    lambda2 (float): Regularization parameter for unlabeled data.
    F (np.ndarray): Matrix F for labeled data.

    Returns:
    np.ndarray: Updated membership matrix for unlabeled data.
    """
    n, c = X_u.shape[0], V.shape[0]
    U_un = np.zeros((c, n))
    Z = np.zeros((c, n))
    T = np.zeros((c, n))

    for r in range(n):
        for i in range(c):
            d_ir = np.linalg.norm(X_u[r] - V[i])
            sum_z = sum(
                s[k] * W[k, r] * F[i, k] + (2 / (s[k] + 1) - 1) * W[k, r] * U_prev_l[i, k] for k in range(W.shape[0]))
            Z[i, r] = lambda2 * sum_z
            sum_t = sum(s[k] * W[k, r] + (2 / (s[k] + 1) - 1) * W[k, r] for k in range(W.shape[0]))
            T[i, r] = d_ir ** 2 + lambda2 * sum_t

    for r in range(n):
        for i in range(c):
            U_un[i, r] = (Z[i, r] + (1 - np.sum(Z[:, r] / T[:, r])) / np.sum(1 / T[:, r])) / T[i, r]

    return U_un


def update_v(X_l, X_un, U_l, U_un, s, lambda1, F):
    """
    Update the cluster centers V.

    Args:
    X_l (np.ndarray): Labeled data points.
    X_un (np.ndarray): Unlabeled data points.
    U_l (np.ndarray): Membership matrix for labeled data.
    U_un (np.ndarray): Membership matrix for unlabeled data.
    s (np.ndarray): Confidence levels.
    lambda1 (float): Regularization parameter for labeled data.
    F (np.ndarray): Matrix F for labeled data.

    Returns:
    np.ndarray: Updated cluster centers.
    """
    X = np.concatenate((X_l, X_un), axis=0)
    U = np.concatenate((U_l, U_un), axis=1)
    V = np.zeros((U.shape[0], X.shape[1]))
    for i in range(U.shape[0]):
        sum1 = np.sum((U[i, :] ** 2)[:, None] * X, axis=0)
        sum2 = np.sum(s[:, None] * (U[i, :len(X_l)] - F[i, :]) ** 2[:, None] * X_l, axis=0)
        sum3 = np.sum(U[i, :] ** 2)
        sum4 = np.sum(s * (U[i, :len(X_l)] - F[i, :]) ** 2)
        V[i, :] = (sum1 + lambda1 * sum2) / (sum3 + lambda1 * sum4)
    return V


def update_s(X_l, W, U_l, U_un, V, F, lambda1, lambda2):
    """
    Update the confidence levels S.

    Args:
    X_l (np.ndarray): Labeled data points.
    W (np.ndarray): Weight matrix.
    U_l (np.ndarray): Membership matrix for labeled data.
    U_un (np.ndarray): Membership matrix for unlabeled data.
    V (np.ndarray): Cluster centers.
    F (np.ndarray): Matrix F for labeled data.
    lambda1 (float): Regularization parameter for labeled data.
    lambda2 (float): Regularization parameter for unlabeled data.

    Returns:
    np.ndarray: Updated confidence levels.
    """
    n = U_l.shape[1]
    Omega = np.zeros(n)
    Delta = np.zeros(n)

    for k in range(n):
        Omega[k] = compute_omega_k(k, W, U_un, U_l, V, lambda2)
        Delta[k] = compute_delta_k(k, X_l, W, U_un, U_l, V, F, lambda1, lambda2)

    return solve_optimization_problem(Omega, Delta)


def compute_omega_k(k, W, U_un, U_l, V, lambda2):
    """
    Compute the omega value for a given data point.

    Args:
    k (int): Index of the data point.
    W (np.ndarray): Weight matrix.
    U_un (np.ndarray): Membership matrix for unlabeled data.
    U_l (np.ndarray): Membership matrix for labeled data.
    V (np.ndarray): Cluster centers.
    lambda2 (float): Regularization parameter for unlabeled data.

    Returns:
    float: Computed omega value.
    """
    n, c = U_un.shape[1], V.shape[0]
    sum2 = sum(W[k, r] * np.sum((U_l[:, k] - U_un[:, r]) ** 2) for r in range(n))
    return 4 * lambda2 * sum2


def compute_delta_k(k, X_l, W, U_un, U_l, V, F, lambda1, lambda2):
    """
    Compute the delta value for a given data point.

    Args:
    k (int): Index of the data point.
    X_l (np.ndarray): Labeled data points.
    W (np.ndarray): Weight matrix.
    U_un (np.ndarray): Membership matrix for unlabeled data.
    U_l (np.ndarray): Membership matrix for labeled data.
    V (np.ndarray): Cluster centers.
    F (np.ndarray): Matrix F for labeled data.
    lambda1 (float): Regularization parameter for labeled data.
    lambda2 (float): Regularization parameter for unlabeled data.

    Returns:
    float: Computed delta value.
    """
    n, c = U_un.shape[1], V.shape[0]
    sum1 = sum(np.linalg.norm(X_l[k] - V[i]) ** 2 * (U_l[i, k] - F[i, k]) ** 2 for i in range(c))
    sum2 = sum(W[k, r] * np.sum((F[:, k] - U_un[:, r]) ** 2) for r in range(n))
    sum3 = sum(W[k, r] * np.sum((U_l[:, k] - U_un[:, r]) ** 2) for r in range(n))
    return lambda1 * sum1 + lambda2 * sum2 - 2 * lambda2 * sum3


def solve_optimization_problem(omega, delta):
    """
    Solve the quadratic programming optimization problem.

    Args:
    omega (np.ndarray): Omega values.
    delta (np.ndarray): Delta values.

    Returns:
    np.ndarray: Optimal confidence levels.
    """
    l = len(omega)
    Q = 2 * cvxopt.matrix(np.diag(omega))
    c = cvxopt.matrix(delta)
    A = cvxopt.matrix(1.0, (1, l))
    b = cvxopt.matrix(1.0)
    G = cvxopt.matrix(np.vstack([-np.eye(l), np.eye(l)]))
    h = cvxopt.matrix(np.hstack([np.zeros(l), np.ones(l)]))
    solution = cvxopt.solvers.qp(Q, c, G, h, A, b)
    return np.array(solution['x']).flatten()


def compute_Ja(W, X_l, X_un, U_un, U_l, V, S, F, lambda1, lambda2):
    """
    Compute the objective function J_a.

    Args:
    W (np.ndarray): Weight matrix.
    X_l (np.ndarray): Labeled data points.
    X_un (np.ndarray): Unlabeled data points.
    U_un (np.ndarray): Membership matrix for unlabeled data.
    U_l (np.ndarray): Membership matrix for labeled data.
    V (np.ndarray): Cluster centers.
    S (np.ndarray): Confidence levels.
    F (np.ndarray): Matrix F for labeled data.
    lambda1 (float): Regularization parameter for labeled data.
    lambda2 (float): Regularization parameter for unlabeled data.

    Returns:
    float: Computed objective function value.
    """
    U = np.concatenate((U_l, U_un), axis=1)
    X = np.concatenate((X_l, X_un), axis=0)
    n, c = U.shape[1], V.shape[0]
    n_l = X_l.shape[0]
    n_un = X_un.shape[0]
    sum1 = sum(np.linalg.norm(X[k] - V[i]) ** 2 * U[i, k] ** 2 for k in range(n) for i in range(c))
    sum2 = sum(
        S[k] * np.linalg.norm(X_l[k] - V[i]) ** 2 * (U[i, k] - F[i, k]) ** 2 for k in range(n_l) for i in range(c))
    sum3 = sum(S[k] * W[k, r] * (F[i, k] - U_un[i, r]) ** 2 for k in range(n_l) for r in range(n_un) for i in range(c))
    sum4 = sum(
        (2 / (S[k] + 1) - 1) * W[k, r] * (U_l[i, k] - U_un[i, r]) ** 2 for k in range(n_l) for r in range(n_un) for i in
        range(c))
    return sum1 + lambda1 * sum2 + lambda2 * (sum3 + sum4)


def compute_distances(data):
    """
    Compute pairwise distances between data points.

    Args:
    data (np.ndarray): Data points.

    Returns:
    list: List of distances and indices.
    """
    return [(sqrt(sum((data[i][k] - data[j][k]) ** 2 for k in range(len(data[i])))), i, j) for i in range(len(data)) for
            j in range(i + 1, len(data))]


def compute_average_distance(data):
    """
    Compute the average distance between data points.

    Args:
    data (np.ndarray): Data points.

    Returns:
    float: Average distance.
    """
    distances = compute_distances(data)
    total_distance = sum(dist[0] for dist in distances)
    return total_distance / len(distances)


def compute_weights(X_l, X_u, sigma, N_pu):
    """
    Compute the weight matrix using a Gaussian kernel.

    Args:
    X_l (np.ndarray): Labeled data points.
    X_u (np.ndarray): Unlabeled data points.
    sigma (float): Standard deviation for the Gaussian kernel.
    N_pu (np.ndarray): Number of nearest neighbors for each labeled instance.

    Returns:
    np.ndarray: Weight matrix.
    """
    l, u = X_l.shape[0], X_u.shape[0]
    W = np.zeros((l, u))
    for i in range(l):
        distances = [distance.euclidean(X_l[i], X_u[j]) for j in range(u)]
        nearest_indices = np.argsort(distances)[:N_pu[i]]
        W[i, nearest_indices] = np.exp(-np.array(distances)[nearest_indices] ** 2 / (sigma ** 2))
    return W


def get_predicted_labels(U):
    """
    Get predicted labels from the membership matrix.

    Args:
    U (np.ndarray): Membership matrix.

    Returns:
    np.ndarray: Predicted labels.
    """
    return np.argmax(U, axis=0)


def compute_CA(y, y_hat):
    """
    Compute the classification accuracy.

    Args:
    y (np.ndarray): True labels.
    y_hat (np.ndarray): Predicted labels.

    Returns:
    float: Classification accuracy.
    """
    n = len(y)
    correct_predictions = np.equal(y, y_hat)
    CA = np.sum(correct_predictions) / n
    print(f'Correct predictions: {np.sum(correct_predictions)}')
    return CA


def AS3FCM(X_l, X_u, Y_l, Y_hat, lambda1, lambda2, sigma, eta, max_iter, m=2):
    """
    Adaptive Semi-Supervised Soft Subspace Clustering with Fuzzy C-Means.

    Args:
    X_l (np.ndarray): Labeled data points.
    X_u (np.ndarray): Unlabeled data points.
    Y_l (np.ndarray): True labels for labeled data.
    Y_hat (np.ndarray): Initial guess for the labels.
    lambda1 (float): Regularization parameter for labeled data.
    lambda2 (float): Regularization parameter for unlabeled data.
    sigma (float): Standard deviation for the Gaussian kernel.
    eta (float): Convergence threshold.
    max_iter (int): Maximum number of iterations.
    m (int): Fuzziness parameter.

    Returns:
    np.ndarray: Final membership matrix.
    """
    N_pu = np.full(X_l.shape[0], 5)
    W = compute_weights(X_l, X_u, sigma, N_pu)
    V = initialize_V(X_l, Y_l, Y_hat)
    U_l = initialize_U(X_l, V, m)
    U_un = initialize_U(X_u, V, m)
    S = initialize_S(X_l.shape[0])
    F = initialize_F(X_l, Y_l, Y_hat)
    previous_Ja = 0

    for t in range(max_iter):
        U_l, U_un = update_u(X_l, X_u, V, W, U_l, U_un, S, lambda1, lambda2, F)
        V = update_v(X_l, X_u, U_l, U_un, S, lambda1, F)
        S = update_s(X_l, W, U_l, U_un, V, F, lambda1, lambda2)
        current_Ja = compute_Ja(W, X_l, X_u, U_un, U_l, V, S, F, lambda1, lambda2)
        if t > 0 and np.abs(current_Ja - previous_Ja) < eta:
            break
        previous_Ja = current_Ja

    return np.concatenate((U_l, U_un), axis=1)


def print_groups(Y_hat, Y_star, filename):
    """
    Print the predicted and true labels to a file.

    Args:
    Y_hat (np.ndarray): True labels.
    Y_star (np.ndarray): Predicted labels.
    filename (str): Output file name.
    """
    combined = np.column_stack((Y_hat, Y_star))
    np.savetxt(filename, combined, fmt='%d')


def AS3FCM_hml():
    """
    Example usage of AS3FCM on a small dataset.
    """
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0], [0, 1], [2, 1], [4, 1], [1, 1]])
    Y_hat = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    X_l, X_u, Y_l = split_data(X, Y_hat, 50, 0)
    lambda1 = 0.1
    lambda2 = 0.1
    sigma = compute_average_distance(np.concatenate((X_l, X_u), axis=0))
    eta = 0.01
    max_iter = 100
    U_star = AS3FCM(X_l, X_u, Y_l, Y_hat, lambda1, lambda2, sigma, eta, max_iter)
    Y_star = get_predicted_labels(U_star)
    CA = compute_CA(Y_hat, Y_star)
    print(f"Classification accuracy: {CA}")


def AS3FCM_prod():
    """
    Example usage of AS3FCM on a real dataset.
    """
    X, Y_hat = extract_features_and_labels('./liver+disorders/bupa.data')
    Y_hat = Y_hat - 1
    X_l, X_u, Y_l = split_data(X, Y_hat, 20, 0)
    lambda1_list = [.001, .01, .1, 1, 10, 100]
    lambda2_list = [.001, .01, .1, 1, 10, 100]
    sigma = compute_average_distance(np.concatenate((X_l, X_u), axis=0))
    eta = 0.01
    max_iter = 100
    results = ""

    for lambda1 in lambda1_list:
        for lambda2 in lambda2_list:
            U_star = AS3FCM(X_l, X_u, Y_l, Y_hat, lambda1, lambda2, sigma, eta, max_iter)
            Y_star = get_predicted_labels(U_star)
            CA = compute_CA(Y_hat, Y_star)
            print(f"Classification accuracy for lambda1={lambda1}, lambda2={lambda2}: {CA}")
            results += f"Classification accuracy for lambda1={lambda1}, lambda2={lambda2}: {CA}\n"

    with open('results#1.txt', 'w') as f:
        f.write(results)


if __name__ == '__main__':
    # AS3FCM_hml()
    AS3FCM_prod()
