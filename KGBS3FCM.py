import argparse
import math
import os
import time

import numpy as np
import cvxopt
import concurrent.futures
import Databases

from math import sqrt
from scipy.spatial import distance
from scipy.spatial.distance import pdist
from Helpers import split_data
from FCMOpt import initialize_U_FCM, initialize_ULabeled, initialize_V, initialize_S, initialize_V_KM, update_U_labeled
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

E = 10e-10


def initialize_F(X_l, Y_l, labels):
    """
    Initialize the matrix F, which indicates the membership of each labeled instance to the clusters.

    Parameters:
    X_l (array): Labeled data instances.
    Y_l (array): Labels for the labeled instances.
    labels (array): Unique labels in the dataset.

    Returns:
    F (array): Initialized matrix F.
    """
    unique_labels = np.unique(labels)  # Unique labels
    c = len(unique_labels)  # Number of unique labels
    n = len(X_l)  # Number of labeled instances
    F = np.zeros((c, n))  # Initialize F with zeros

    # Create a mapping of original labels to 0-indexed labels
    label_mapping = {label: index for index, label in enumerate(unique_labels)}

    for k in range(n):
        original_label = Y_l[k]  # Original label of instance k
        i = label_mapping[original_label]  # Mapped label of instance k
        F[i, k] = 1  # Set f_ik = 1

    return F


def update_u(X_l, X_un, V, W, U_prev_l, U_prev_un, s, lambda1, lambda2, F):
    """
    Update the membership matrix U for both labeled and unlabeled data.

    Parameters:
    X_l (array): Labeled data instances.
    X_un (array): Unlabeled data instances.
    V (array): Cluster centers.
    W (array): Weight matrix.
    U_prev_l (array): Previous membership matrix for labeled data.
    U_prev_un (array): Previous membership matrix for unlabeled data.
    s (array): Safety degrees for labeled data.
    lambda1 (float): Regularization parameter.
    lambda2 (float): Regularization parameter.
    F (array): Initialized matrix F.

    Returns:
    U_l, U_un (arrays): Updated membership matrices for labeled and unlabeled data.
    """
    U_l = update_U_labeled(X_l, V, W, U_prev_un, s, lambda1, lambda2, F)
    U_un = update_U_unlabeled(X_un, V, W, U_prev_l, s, lambda2, F)
    return U_l, U_un


def update_U_unlabeled(X_u, V, W, U_prev_l, s, lambda2, F):
    """
    Update the membership matrix U for unlabeled data.

    Parameters:
    X_u (array): Unlabeled data instances.
    V (array): Cluster centers.
    W (array): Weight matrix.
    U_prev_l (array): Previous membership matrix for labeled data.
    s (array): Safety degrees for labeled data.
    lambda2 (float): Regularization parameter.
    F (array): Initialized matrix F.

    Returns:
    U_un (array): Updated membership matrix for unlabeled data.
    """
    n, c = X_u.shape[0], V.shape[0]  # n data points, c clusters
    U_un = np.zeros((c, n))  # initialize the U matrix

    # compute matrix Z (c  x n) and T (c x n)
    Z = np.zeros((c, n))
    T = np.zeros((c, n))

    for r in range(n):
        for i in range(c):  # Loop over clusters
            sum_z = 0
            sum_t = 0
            d_ir = np.linalg.norm(X_u[r] - V[i])  # Euclidean distance from X[k] to V[i]
            for k in range(W.shape[0]):  # Loop over labeled data points
                sum_z += s[k] * W[k, r] * F[i, k] + (2 / (s[k] + 1) - 1) * W[k, r] * U_prev_l[i, k]
                Z[i, r] = lambda2 * sum_z

                sum_t += s[k] * W[k, r] + (2 / (s[k] + 1) - 1) * W[k, r]
                T[i, r] = d_ir ** 2 + lambda2 * sum_t

    for r in range(n):  # Loop over data points
        for i in range(c):  # Loop over clusters
            U_un[i, r] = (Z[i, r] + (1 - np.sum(Z[:, r] / T[:, r])) / np.sum(1 / T[:, r])) / T[i, r]

    return U_un


def update_v(X_l, X_un, U_l, U_un, s, lambda1, F):
    """
    Update the cluster centers V.

    Parameters:
    X_l (array): Labeled data instances.
    X_un (array): Unlabeled data instances.
    U_l (array): Membership matrix for labeled data.
    U_un (array): Membership matrix for unlabeled data.
    s (array): Safety degrees for labeled data.
    lambda1 (float): Regularization parameter.
    F (array): Initialized matrix F.

    Returns:
    V (array): Updated cluster centers.
    """
    X = np.concatenate((X_l, X_un), axis=0)
    U = np.concatenate((U_l, U_un), axis=1)
    V = np.zeros((U.shape[0], X.shape[1]))  # Initialize the cluster centers matrix V
    for i in range(U.shape[0]):  # Loop over clusters
        sum1 = 0
        sum2 = 0
        sum4 = 0
        for k in range(X.shape[0]):  # Loop over data points
            sum1 += (U[i, k] ** 2) * X[k, :]
        for k in range(X_l.shape[0]):
            sum2 += s[k] * (U[i, k] - F[i, k]) ** 2 * X[k, :]
            sum4 += s[k] * (U[i, k] - F[i, k]) ** 2
        sum3 = np.sum(U[i, :] ** 2)
        V[i, :] = (sum1 + lambda1 * sum2) / (sum3 + lambda1 * sum4)
    return V


def update_s(X_l, W, U_l, U_un, V, F, lambda1, lambda2):
    """
    Update the safety degrees S for labeled data.

    Parameters:
    X_l (array): Labeled data instances.
    W (array): Weight matrix.
    U_l (array): Membership matrix for labeled data.
    U_un (array): Membership matrix for unlabeled data.
    V (array): Cluster centers.
    F (array): Initialized matrix F.
    lambda1 (float): Regularization parameter.
    lambda2 (float): Regularization parameter.

    Returns:
    S (array): Updated safety degrees.
    """
    n = U_l.shape[1]  # Number of data points
    # Computing Omega and Delta Array
    Omega = np.zeros(n)
    Delta = np.zeros(n)

    for k in range(n):
        Omega[k] = compute_omega_k(k, W, U_un, U_l, V, lambda2)
        Delta[k] = compute_delta_k(k, X_l, W, U_un, U_l, V, F, lambda1, lambda2)

    return solve_optimization_problem(Omega, Delta)


def compute_omega_k(k, W, U_un, U_l, V, lambda2):
    """
    Compute the Omega_k value for a given data point.

    Parameters:
    k (int): Index of the data point.
    W (array): Weight matrix.
    U_un (array): Membership matrix for unlabeled data.
    U_l (array): Membership matrix for labeled data.
    V (array): Cluster centers.
    lambda2 (float): Regularization parameter.

    Returns:
    Omega_k (float): Computed Omega_k value.
    """
    n, c = U_un.shape[1], V.shape[0]  # n data unlabeled points, c clusters
    sum2 = 0
    for r in range(n):
        sum1 = 0
        for i in range(c):
            sum1 += (U_l[i, k] - U_un[i, r]) ** 2
        sum2 += W[k, r] * sum1
    return 4 * lambda2 * sum2  # return Omega_k


def compute_delta_k(k, X_l, W, U_un, U_l, V, F, lambda1, lambda2):
    """
    Compute the Delta_k value for a given data point.

    Parameters:
    k (int): Index of the data point.
    X_l (array): Labeled data instances.
    W (array): Weight matrix.
    U_un (array): Membership matrix for unlabeled data.
    U_l (array): Membership matrix for labeled data.
    V (array): Cluster centers.
    F (array): Initialized matrix F.
    lambda1 (float): Regularization parameter.
    lambda2 (float): Regularization parameter.

    Returns:
    Delta_k (float): Computed Delta_k value.
    """
    n, c = U_un.shape[1], V.shape[0]  # n data unlabeled points, c clusters
    sum1 = 0
    for i in range(c):
        d_ik = max(E, np.linalg.norm(X_l[k] - V[i]))  # Euclidean distance from X[k] to V[i]
        sum1 += d_ik ** 2 * (U_l[i, k] - F[i, k]) ** 2

    sum2: int = 0
    for r in range(n):
        internal_sum = 0
        for i in range(c):
            internal_sum += (F[i, k] - U_un[i, r]) ** 2
        sum2 += W[k, r] * internal_sum

    sum3 = 0
    for r in range(n):
        internal_sum = 0
        for i in range(c):
            internal_sum += (U_l[i, k] - U_un[i, r]) ** 2
        sum3 += W[k, r] * internal_sum

    return lambda1 * sum1 + lambda2 * sum2 - 2 * lambda2 * sum3  # return Delta_k


def solve_optimization_problem(omega, delta):
    """
    Solve the quadratic programming problem to optimize the safety degrees S.

    Parameters:
    omega (array): Omega values for the optimization problem.
    delta (array): Delta values for the optimization problem.

    Returns:
    s_k_optimal (array): Optimized values for the safety degrees S.
    """
    l = len(omega)  # Number of data points
    # Convert to cvxopt matrix format
    Q = 2 * cvxopt.matrix(np.diag(omega))  # Multiply by 2 because QP expects 0.5 * x^T * Q * x
    c = cvxopt.matrix(delta)

    # Equality constraint A * x = b
    A = cvxopt.matrix(1.0, (1, l))  # Constraint sum(s_k) = 1
    b = cvxopt.matrix(1.0)

    # Inequality constraints G * x <= h
    # This represents 0 <= s_k <= 1
    G = cvxopt.matrix(np.vstack([-np.eye(l), np.eye(l)]))
    h = cvxopt.matrix(np.hstack([np.zeros(l), np.ones(l)]))

    # Solve the quadratic programming problem
    solution = cvxopt.solvers.qp(Q, c, G, h, A, b)

    # Extract the optimal values for s_k
    s_k_optimal = np.array(solution['x']).flatten()  # Convert from cvxopt matrix to numpy array

    # print("Optimal values for s_k:", sum(s_k_optimal))
    return s_k_optimal


def compute_Ja(W, X_l, X_un, U_un, U_l, V, S, F, lambda1, lambda2):
    """
    Compute the objective function Ja.

    Parameters:
    W (array): Weight matrix.
    X_l (array): Labeled data instances.
    X_un (array): Unlabeled data instances.
    U_un (array): Membership matrix for unlabeled data.
    U_l (array): Membership matrix for labeled data.
    V (array): Cluster centers.
    S (array): Safety degrees for labeled data.
    F (array): Initialized matrix F.
    lambda1 (float): Regularization parameter.
    lambda2 (float): Regularization parameter.

    Returns:
    Ja (float): Computed value of the objective function.
    """
    U = np.concatenate((U_l, U_un), axis=1)
    X = np.concatenate((X_l, X_un), axis=0)
    n, c = U.shape[1], V.shape[0]  # n data points, c clusters
    n_l = X_l.shape[0]  # Number of labeled data points
    n_un = X_un.shape[0]  # Number of unlabeled data points
    sum1 = 0
    for k in range(n):
        for i in range(c):
            d_ik = np.linalg.norm(X[k] - V[i])
            sum1 += d_ik ** 2 * U[i, k] ** 2

    sum2 = 0
    for k in range(n_l):
        for i in range(c):
            d_ik = np.linalg.norm(X_l[k] - V[i])
            sum2 += S[k] * (U[i, k] - F[i, k]) ** 2 * d_ik ** 2

    sum3 = 0
    for k in range(n_l):
        internal_sum1 = 0
        internal_sum2 = 0
        for r in range(n_un):
            for i in range(c):
                internal_sum1 += S[k] * W[k, r] * (F[i, k] - U_un[i, r]) ** 2

        for r in range(n_un):
            for i in range(c):
                internal_sum2 += (2 / (S[k] + 1) - 1) * W[k, r] * (U_l[i, k] - U_un[i, r]) ** 2

        sum3 += internal_sum1 + internal_sum2

    return  sum1 + lambda1 * sum2 + lambda2 * sum3


def compute_distances(data):
    """
    Compute the pairwise distances between data points using a Euclidean metric.

    Parameters:
    data (array): Input data points.

    Returns:
    pairwise_distances (array): Pairwise distances between data points.
    """
    pairwise_distances = pdist(data, metric='euclidean')
    return pairwise_distances


def compute_average_distance(data):
    """
    Compute the average pairwise distance between data points.

    Parameters:
    data (array): Input data points.

    Returns:
    average_distance (float): Average pairwise distance.
    """
    distances = compute_distances(data)
    total_distance = sum(dist for dist in distances)
    average_distance = total_distance / len(distances)
    return average_distance


def compute_weights(X_l, X_u, sigma, N_pu):
    """
    Compute the weight matrix W using a Gaussian kernel.

    Parameters:
    X_l (array): Labeled data instances.
    X_u (array): Unlabeled data instances.
    sigma (float): Standard deviation for the Gaussian kernel.
    N_pu (array): Number of nearest neighbors for each labeled instance.

    Returns:
    W (array): Computed weight matrix.
    """
    l, u = X_l.shape[0], X_u.shape[0]  # Number of labeled and unlabeled instances
    W = np.zeros((l, u))  # Initialize the weight matrix W

    # Compute the weight matrix using a Gaussian kernel
    for i in range(l):
        distances = [distance.euclidean(X_l[i], X_u[j]) for j in range(u)]
        nearest_indices = np.argsort(distances)[:N_pu[i]]
        for j in range(u):
            if j in nearest_indices:
                W[i, j] = np.exp(-distances[j] ** 2 / (sigma ** 2))
            else:
                W[i, j] = 0

    return W


def get_predicted_labels(U):
    """
    Get the predicted labels from the membership matrix U.

    Parameters:
    U (array): Membership matrix.

    Returns:
    Y (array): Predicted labels.
    """
    Y = np.argmax(U, axis=0)
    return Y


def compute_CA(y, y_hat):
    """
    Compute the Classification Accuracy (CA).

    Parameters:
    y (array): True labels.
    y_hat (array): Predicted labels.

    Returns:
    CA (float): Classification accuracy.
    """
    n = len(y)  # Number of instances
    correct_predictions = np.equal(y, y_hat)  # Compare true and predicted labels
    print(f'Correct predictions:  {np.sum(correct_predictions)}')
    CA = np.sum(correct_predictions) / n  # Compute classification accuracy
    return CA


def compute_safety_degrees(X_l, X_u, U_un, F, N_pu):
    """
    Compute the safety degrees S for labeled data based on local consistency.

    Parameters:
    X_l (array): Labeled data instances.
    X_u (array): Unlabeled data instances.
    U_un (array): Membership matrix for unlabeled data.
    F (array): Initialized matrix F.
    N_pu (array): Number of nearest neighbors for each labeled instance.

    Returns:
    S (array): Computed safety degrees.
    """
    X = np.concatenate((X_l, X_u), axis=0)
    num_labeled = X_l.shape[0]
    S = np.zeros(num_labeled)

    for i in range(num_labeled):
        k = N_pu[i]
        if k == 0:
            S[i] = 1
            continue

        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)
        _, neighbors = knn.kneighbors([X_l[i]])
        neighbors = neighbors[0]

        # Filter out labeled neighbors
        unlabeled_neighbors = [n - num_labeled for n in neighbors if n >= num_labeled]

        if len(unlabeled_neighbors) == 0:
            S[i] = 1
            continue

        inconsistency = 0
        for neighbor in unlabeled_neighbors:
            inconsistency += np.linalg.norm(U_un[:, neighbor] - F[:, i])

        S[i] = 1 / (1 + inconsistency / len(unlabeled_neighbors))

    return S


def compute_N_pu(X_l, k, min_value=0, max_value=math.inf):
    """
    Compute the number of nearest neighbors (N_pu) for each labeled instance based on average distance.

    Parameters:
    X_l (array): Labeled data instances.
    k (int): Number of neighbors for KNN.
    min_value (int): Minimum value for normalization.
    max_value (int): Maximum value for normalization.

    Returns:
    N_pu2 (array): Computed number of nearest neighbors for each labeled instance.
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_l, np.zeros(X_l.shape[0]))
    distances, _ = knn.kneighbors(X_l)
    N_pu = np.round(np.mean(distances, axis=1)).astype(int)

    # COMPUTE N_pu based on the average distance (proxy for density estimation) of the labeled data
    normalized_k = min_value + (max_value - min_value) * (N_pu - np.min(N_pu)) / (np.max(N_pu) - np.min(N_pu))
    N_pu2 = np.clip(normalized_k, min_value, max_value).astype(int)

    return N_pu2


def step(x, a):
    """
    Step function for activation.

    Parameters:
    x (float): Input value.
    a (float): Activation threshold.

    Returns:
    (int): Step function result (1 if x > a, else 0).
    """
    if x > a:
        return 1
    return 0


def X_S3FCM(X_l, X_u, V, U_l, U_un, S, F, N_pu, W, lambda1, lambda2, activation_threshold, eta, max_iter, gb):
    """
    Main iteration loop for the X_S3FCM algorithm.

    Parameters:
    X_l (array): Labeled data instances.
    X_u (array): Unlabeled data instances.
    V (array): Cluster centers.
    U_l (array): Membership matrix for labeled data.
    U_un (array): Membership matrix for unlabeled data.
    S (array): Safety degrees for labeled data.
    F (array): Initialized matrix F.
    N_pu (array): Number of nearest neighbors for each labeled instance.
    W (array): Weight matrix.
    lambda1 (float): Regularization parameter.
    lambda2 (float): Regularization parameter.
    activation_threshold (float): Threshold for the activation function.
    eta (float): Convergence threshold.
    max_iter (int): Maximum number of iterations.
    gb (bool): Boolean flag indicating if graph-based updating is used.

    Returns:
    U (array): Final partition matrix.
    """
    previous_Ja = 0
    # Main iteration loop
    t = 0

    lambda2_orig = lambda2
    for t in range(max_iter):
        # Step 4: Update u
        U_l, U_un = update_u(X_l, X_u, V, W, U_l, U_un, S, lambda1, lambda2, F)

        # Step 5: Update v
        V = update_v(X_l, X_u, U_l, U_un, S, lambda1, F)

        # Step 6: Update s
        if gb:
            S = compute_safety_degrees(X_l, X_u, U_un, F, N_pu)  # Update S based on local consistency
            # print(f"Safety degrees: {S}")
            # update lambda2 based on the activation function, this represents the influence of labeled regularization on unlabeled data
            lambda2 = lambda2_orig * (lambda1 ** step(np.mean(S), activation_threshold))
            #print(f"Lambda2: {lambda2} and mean(S): {np.mean(S)}")

        else:
            S = update_s(X_l, W, U_l, U_un, V, F, lambda1, lambda2)

        # Step 7: Compute J_a
        current_Ja = compute_Ja(W, X_l, X_u, U_un, U_l, V, S, F, lambda1, lambda2)

        # Step 8: Check convergence
        if t > 0 and np.abs(current_Ja - previous_Ja) < eta:
            break

        previous_Ja = current_Ja
    print("Stoppped at ITER: " + str(t) + " with AvgS: " + str(np.mean(S)))

    # Return the final partition matrix
    return np.concatenate((U_l, U_un), axis=1)


def X3FCM_process_combination(lambda1, lambda2, X_l, X_u, V, U_l, U_un, S, F, N_pu, W, activation, eta, max_iter, gb, Y_hat):
    """
    Process a single combination of lambda1 and lambda2 for the X3FCM algorithm.

    Parameters:
    lambda1 (float): Regularization parameter.
    lambda2 (float): Regularization parameter.
    X_l (array): Labeled data instances.
    X_u (array): Unlabeled data instances.
    V (array): Cluster centers.
    U_l (array): Membership matrix for labeled data.
    U_un (array): Membership matrix for unlabeled data.
    S (array): Safety degrees for labeled data.
    F (array): Initialized matrix F.
    N_pu (array): Number of nearest neighbors for each labeled instance.
    W (array): Weight matrix.
    activation (float): Activation threshold.
    eta (float): Convergence threshold.
    max_iter (int): Maximum number of iterations.
    gb (bool): Boolean flag indicating if graph-based updating is used.
    Y_hat (array): True labels.

    Returns:
    CA (float): Classification accuracy.
    result_str (str): Result string containing the classification accuracy.
    """
    # Compute Performance Metrics
    U_star = X_S3FCM(X_l, X_u, V, U_l, U_un, S, F, N_pu, W, lambda1, lambda2, activation, eta, max_iter, gb)
    Y_star = get_predicted_labels(U_star)

    # get overall accuracy
    CA = compute_CA(Y_hat, Y_star)

    result_str = f"Classification accuracy for lambda1={lambda1}, lambda2={lambda2}: {CA}\n"
    print(result_str)
    return CA, result_str


def X3FCM_prod(trial, X, Y_hat, mislabeling_percentage, label, gb=True):
    """
    Run the X3FCM algorithm for a given trial and compute the classification accuracy.

    Parameters:
    trial (int): Trial number.
    X (array): Input data instances.
    Y_hat (array): True labels.
    mislabeling_percentage (float): Percentage of mislabeling in the data.
    label (str): Label of the dataset.
    gb (bool): Boolean flag indicating if graph-based updating is used.

    Returns:
    max_accuracy (float): Maximum classification accuracy.
    """
    # Shuffle the data, for distinct results
    X_l, X_u, Y_l = split_data(X, Y_hat, 20, mislabeling_percentage)
    # Example usage
    lambda1_list = [.001, .01, .1, 1, 10, 100]
    lambda2_list = [.001, .01, .1, 1, 10, 100]
    eta = 0.0001
    max_iter = 100

    # Step 1: Constructing the Graph Initialize W if necessary
    if gb:  # Number of nearest neighbors for each labeled instance
        N_pu = compute_N_pu(X_l, 5, 10, sqrt(len(Y_hat)))
    else:
        N_pu = np.full(X_l.shape[0], 5)
    print(f'N_pu: {N_pu}')

    sigma = compute_average_distance(np.concatenate((X_l, X_u), axis=0))
    W = compute_weights(X_l, X_u, sigma, N_pu)

    # Step 2: Initialize cluster variables
    V = initialize_V(X_l, Y_l, Y_hat)
    # V = initialize_V_KM(X_l, Y_hat) # this is not helping much

    U_l = initialize_ULabeled(V.shape[0], Y_l)
    U_un = initialize_U_FCM(X_u, V, 2)
    S = initialize_S(X_l.shape[0])
    F = initialize_F(X_l, Y_l, Y_hat)

    results = ""
    accuracies = []
    activation = .6

    # Create a list of all combinations of lambda1 and lambda2
    combinations = [(lambda1, lambda2) for lambda1 in lambda1_list for lambda2 in lambda2_list]

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(X3FCM_process_combination, lambda1, lambda2, X_l, X_u, V, U_l, U_un, S, F, N_pu, W, activation,
                            eta, max_iter, gb, Y_hat) for lambda1, lambda2 in combinations]
        for future in concurrent.futures.as_completed(futures):
            CA, result_str = future.result()
            accuracies.append(CA)
            results += result_str
    if gb:
        prefix = 'GB'
    else:
        prefix = 'AS'

    # Write the results to a file - append mislabeling_percentage to the filename
    with open(prefix +'_'+ label +'_R' + str(mislabeling_percentage) + '_T' + str(trial) + '.txt', 'w') as f:
        f.write(results)

    # Return the best accuracy
    return max(accuracies)


def process_percentage(db_name, mislabeling_percentage, num_runs, gb):
    """
    Process a given mislabeling percentage for a specified number of runs and compute the average accuracy.

    Parameters:
    db_name (str): Name of the database.
    mislabeling_percentage (float): Percentage of mislabeling in the data.
    num_runs (int): Number of runs to perform.
    gb (bool): Boolean flag indicating if graph-based updating is used.

    Returns:
    result_str (str): Result string containing the average accuracy.
    """
    total_accuracy = 0
    label = ""
    for i in range(num_runs):
        X, Y_hat, label = Databases.select_database(db_name)

        print(f'[UPDATE] Running trial {i} for {label} with {mislabeling_percentage}% mislabeling...')
        total_accuracy += X3FCM_prod(i, X, Y_hat, mislabeling_percentage, label, gb)
    average_accuracy = total_accuracy / num_runs

    if gb:
        prefix = 'GB'
    else:
        prefix = 'AS'

    result_str = f"[{label}-{prefix}-{mislabeling_percentage}] Average accuracy over {num_runs} runs: {average_accuracy}\n"
    return result_str


def main(db_name, percentages, num_runs):
    """
    Main function to run the X3FCM algorithm for different mislabeling percentages and compute the average accuracy.

    Parameters:
    db_name (str): Name of the database.
    percentages (list): List of mislabeling percentages to process.
    num_runs (int): Number of runs to perform for each percentage.
    """
    # Record the start time
    start_time = time.time()
    # Run it X times and get the average accuracy
    gb = True

    if percentages is None:
        percentages = [0, 5, 10, 15, 20, 25, 30]

    result_str = ""

    # Determine the number of CPU cores
    num_cores = os.cpu_count()
    print(f"Number of CPU cores: {num_cores}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores//2) as executor:
        futures = [executor.submit(process_percentage, db_name, mislabeling_percentage, num_runs, gb) for mislabeling_percentage
                   in percentages]
        for future in concurrent.futures.as_completed(futures):
            result_str += future.result()
            print(future.result())

    print(result_str)
    # Record the end time
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Done: {elapsed_time / 60} minutes")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="XS3FCM script.")
    parser.add_argument('--db', type=str, required=True, help='DB: {BUPA, DERMATOLOGY, DIABETES, HEART, WAVEFORM, WDBC, GAUSS50, GAUSS50x}')
    parser.add_argument('--pm', nargs='+', type=int, required=False, default=None, help='Percentage of mislabeled')
    parser.add_argument('--nr', type=int, required=False, default=20, help='Number of executions required')
    args = parser.parse_args()

    main(args.db, args.pm, args.nr)


