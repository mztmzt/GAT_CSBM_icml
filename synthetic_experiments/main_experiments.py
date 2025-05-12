import numpy as np
from functions import myGAT, calculate_accuracy
from scipy.special import comb

n = 3000
n1 = n2 = round(n/2)
gt1 = np.ones((n1))
gt2 = np.ones((n2)) * (-1)
gt = np.concatenate((gt1, gt2))

sigma = 10
mu = 2 * sigma * np.sqrt(np.log(n)) / np.power(n, 1/3)

p_in = 3 * np.square(np.log(n)) / n  # Intra-community connection probability
p_out = 2 * np.square(np.log(n)) / n  # Inter-community connection probability


T = [0, 0.5, 0.5, 5]  # Coefficients for the attention mechanism
L = 4  # Number of attention layers

mu_data = np.empty(0, dtype=int)
acc_data = np.empty(0, dtype=int)
for j in range(100):
    mu = 0.1 * (j+1) * sigma * np.sqrt(np.log(n)) / np.power(n, 1/3)

    trials = 100
    acc = 0
    for i in range(trials):
        """
        Generate adjacency matrix A
        """
        # Create an adjacency matrix for the stochastic block model
        A = np.zeros((n1 + n2, n1 + n2))  # Initialize an all-zero adjacency matrix

        # Intra-community connections
        A[:n1, :n1] = np.random.rand(n1, n1) < p_in
        A[n1:, n1:] = np.random.rand(n2, n2) < p_in

        # Inter-community connections
        A[:n1, n1:] = np.random.rand(n1, n2) < p_out
        A[n1:, :n1] = np.random.rand(n2, n1) < p_out

        # The adjacency matrix for an undirected graph should be symmetric
        A = A + A.T

        # Convert probabilities to 0 and 1 to represent the existence of connections
        A = (A > 0).astype(int)

        np.fill_diagonal(A, 0)

        """
        Generate feature matrix X
        """
        X1 = np.random.normal(mu, sigma, n1)
        X2 = np.random.normal(-1 * mu, sigma, n2)

        X = np.concatenate((X1, X2))
        X = np.reshape(X, (n, 1))

        """
        Use graph neural network with attention mechanism
        """
        X_temp = X
        # print(np.mean(X[:n1, :], axis=0), np.var(X[:n1, :], axis=0))

        label = np.sign(X)
        label = np.reshape(label, (n))
        # acc_ori = calculate_accuracy(gt, label)
        # print("acc_ori", acc_ori)

        for t in T:
            X_out = myGAT(A, X_temp, n1, n2, t)
            X_temp = X_out
            label = np.sign(X_out)
            label = np.reshape(label, (n))
        acc += calculate_accuracy(gt, label)
    acc = acc / trials
    mu_data = np.append(mu_data, mu)
    acc_data = np.append(acc_data, acc)
np.savetxt("./100trials_data/mu_data_GCN_GAT.txt", mu_data)
np.savetxt("./100trials_data/acc_data_GCN_GAT.txt", acc_data)
print(mu_data, acc_data)
