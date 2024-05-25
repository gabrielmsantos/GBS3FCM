import numpy as np
import cvxopt
from math import sqrt
from scipy.spatial import distance
from Helpers import split_data, extract_features_and_labels
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph, NearestNeighbors

E = 10e-10


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


def initialize_U(X, V, m=2):
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


def initialize_S(size):
    return np.full(size, 0.5)


def initialize_F(X_l, Y_l, labels):
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
    U_l = update_U_labeled(X_l, V, W, U_prev_un, s, lambda1, lambda2, F)
    U_un = update_U_unlabeled(X_un, V, W, U_prev_l, s, lambda2, F)
    return U_l, U_un


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


def update_U_unlabeled(X_u, V, W, U_prev_l, s, lambda2, F):
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
    n = U_l.shape[1]  # Number of data points
    # Computing Omega and Delta Array
    Omega = np.zeros(n)
    Delta = np.zeros(n)

    for k in range(n):
        Omega[k] = compute_omega_k(k, W, U_un, U_l, V, lambda2)
        Delta[k] = compute_delta_k(k, X_l, W, U_un, U_l, V, F, lambda1, lambda2)

    return solve_optimization_problem(Omega, Delta)


def compute_omega_k(k, W, U_un, U_l, V, lambda2):
    n, c = U_un.shape[1], V.shape[0]  # n data unlabeled points, c clusters
    sum2 = 0
    for r in range(n):
        sum1 = 0
        for i in range(c):
            sum1 += (U_l[i, k] - U_un[i, r]) ** 2
        sum2 += W[k, r] * sum1
    return 4 * lambda2 * sum2  # return Omega_k


def compute_delta_k(k, X_l, W, U_un, U_l, V, F, lambda1, lambda2):
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

    return sum1 + lambda1 * sum2 + lambda2 * sum3


def compute_distances(data):
    l_distances = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            distance = sqrt(sum((data[i][k] - data[j][k]) ** 2 for k in range(len(data[i]))))
            l_distances.append((distance, i, j))
    return l_distances


def compute_average_distance(data):
    distances = compute_distances(data)
    total_distance = sum(dist[0] for dist in distances)
    average_distance = total_distance / len(distances)
    return average_distance


def compute_weights(X_l, X_u, sigma, N_pu):
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
    Y = np.argmax(U, axis=0)
    return Y


def compute_CA(y, y_hat):
    n = len(y)  # Number of instances
    correct_predictions = np.equal(y, y_hat)  # Compare true and predicted labels
    print(f'Correct predictions:  {np.sum(correct_predictions)}')
    CA = np.sum(correct_predictions) / n  # Compute classification accuracy
    return CA


def construct_graph(X_l, X_u, k):
    X = np.concatenate((X_l, X_u), axis=0)
    connectivity = kneighbors_graph(X, n_neighbors=k, include_self=False)
    return connectivity.toarray()


def compute_safety_degrees_OLD(X_l, X_u, U_l, U_un, F, k):
    X = np.concatenate((X_l, X_u), axis=0)
    connectivity = kneighbors_graph(X, n_neighbors=k, include_self=False).toarray()
    S = np.zeros(X_l.shape[0])

    for i in range(X_l.shape[0]):
        neighbors = np.where(connectivity[i])[0]
        labeled_neighbors = neighbors[neighbors < X_l.shape[0]]
        unlabeled_neighbors = neighbors[neighbors >= X_l.shape[0]] - X_l.shape[0]

        if len(unlabeled_neighbors) == 0:
            S[i] = 1
            continue

        consistency = 0
        for neighbor in unlabeled_neighbors:
            consistency += np.linalg.norm(U_un[:, neighbor] - F[:, i])

        S[i] = 1 / (1 + consistency / len(unlabeled_neighbors))

    return S


def compute_safety_degrees(X_l, X_u, U_un, F, N_pu):
    X = np.concatenate((X_l, X_u), axis=0)
    num_labeled = X_l.shape[0]
    num_unlabeled = X_u.shape[0]
    S = np.zeros(num_labeled)

    for i in range(num_labeled):
        k = N_pu[i]
        if k == 0:
            S[i] = 1
            continue

        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)
        distances, neighbors = knn.kneighbors([X_l[i]])
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


def compute_N_pu(X_l, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_l, np.zeros(X_l.shape[0]))
    distances, _ = knn.kneighbors(X_l)
    N_pu = np.round(np.mean(distances, axis=1)).astype(int)
    return N_pu


def AS3FCM(X_l, X_u, Y_l, Y_hat, lambda1, lambda2, sigma, eta, max_iter, m=2):
    # Step 1: Constructing the Graph Initialize W if necessary
    # @TODO:Improve this initialization: KNN Density Estimation
    # N_pu = np.full(X_l.shape[0], 5)  # Number of nearest neighbors for each labeled instance
    N_pu = compute_N_pu(X_l, 5)
    print(f'N_pu: {N_pu}')
    W = compute_weights(X_l, X_u, sigma, N_pu)

    # Step 2: Initialize cluster variables
    # V = initialize_V(X_l, Y_l)
    V = initialize_V(X_l, Y_l, Y_hat)
    U_l = initialize_U(X_l, V, m)
    U_un = initialize_U(X_u, V, m)
    S = initialize_S(X_l.shape[0])
    F = initialize_F(X_l, Y_l, Y_hat)

    # Concatenating labeled and unlabeled data
    # X = np.concatenate((X_l, X_u), axis=0)
    previous_Ja = 0
    # Main iteration loop
    for t in range(max_iter):
        # Step 4: Update u
        U_l, U_un = update_u(X_l, X_u, V, W, U_l, U_un, S, lambda1, lambda2, F)

        # Step 5: Update v
        V = update_v(X_l, X_u, U_l, U_un, S, lambda1, F)

        # Step 6: Update s
        S = compute_safety_degrees(X_l, X_u, U_un, F, N_pu)  # Update S based on local consistency
        # S = update_s(X_l, W, U_l, U_un, V, F, lambda1, lambda2)

        # Step 7: Compute J_a
        current_Ja = compute_Ja(W, X_l, X_u, U_un, U_l, V, S, F, lambda1, lambda2)

        # Step 8: Check convergence
        if t > 0 and np.abs(current_Ja - previous_Ja) < eta:
            break

        previous_Ja = current_Ja

    # Return the final partition matrix
    return np.concatenate((U_l, U_un), axis=1)


def print_groups(Y_hat, Y_star, filename):
    # Assuming Y_hat and Y_star are numpy arrays
    combined = np.column_stack((Y_hat, Y_star))
    # Save to a file
    np.savetxt('output.txt', combined, fmt='%d')


def AS3FCM_hml():
    # Step 1: Create a small dataset with 10 data points and known labels
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0], [0, 1], [2, 1], [4, 1], [1, 1]])
    Y_hat = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

    # Step 2: Split the dataset into labeled and unlabeled data
    X_l, X_u, Y_l = split_data(X, Y_hat, 50, 0)  # 50% of the data is labeled

    # Parameters for AS3FCM
    lambda1 = 0.1
    lambda2 = 0.1
    sigma = compute_average_distance(np.concatenate((X_l, X_u), axis=0))
    eta = 0.01
    max_iter = 100

    # Step 3: Run the AS3FCM function on the dataset
    U_star = AS3FCM(X_l, X_u, Y_l, Y_hat, lambda1, lambda2, sigma, eta, max_iter)

    # Step 4: Get the predicted labels
    Y_star = get_predicted_labels(U_star)

    # Step 5: Compare the predicted labels with the known labels and calculate the classification accuracy
    CA = compute_CA(Y_hat, Y_star)
    print(f"Classification accuracy: {CA}")


def AS3FCM_prod(trial):
    # Shuffle the data, for distinct results
    X, Y_hat = extract_features_and_labels('./dermatology/dermatology.data', shuffle_data=True)
    Y_hat = Y_hat - 1
    mislabeling_percentage = 30
    X_l, X_u, Y_l = split_data(X, Y_hat, 20, mislabeling_percentage)
    # Example usage
    lambda1_list = [.001, .01, .1, 1, 10, 100]
    lambda2_list = [.001, .01, .1, 1, 10, 100]
    sigma = compute_average_distance(np.concatenate((X_l, X_u), axis=0))
    eta = 0.00001
    max_iter = 100

    results = ""
    accuracies = []
    # All combinations of lambda1 and lambda2
    for lambda1 in lambda1_list:
        for lambda2 in lambda2_list:
            # Compute Performance Metrics
            U_star = AS3FCM(X_l, X_u, Y_l, Y_hat, lambda1, lambda2, sigma, eta, max_iter)
            Y_star = get_predicted_labels(U_star)
            CA = compute_CA(Y_hat, Y_star)
            accuracies.append(CA)
            print(f"Classification accuracy for lambda1={lambda1}, lambda2={lambda2}: {CA}")
            results += f"Classification accuracy for lambda1={lambda1}, lambda2={lambda2}: {CA}\n"

    # Write the results to a file - append mislabeling_percentage to the filename
    with open('results_' + str(mislabeling_percentage) + '_T' + str(trial) + '.txt', 'w') as f:
        f.write(results)

    # Return the best accuracy
    return max(accuracies)


if __name__ == '__main__':
    # AS3FCM_hml()
    # Run it X times and get the average accuracy
    num_runs = 20
    total_accuracy = 0
    for i in range(num_runs):
        total_accuracy += AS3FCM_prod(i)
    average_accuracy = total_accuracy / num_runs
    print(f"Average accuracy over {num_runs} runs: {average_accuracy}")
    print("Done")