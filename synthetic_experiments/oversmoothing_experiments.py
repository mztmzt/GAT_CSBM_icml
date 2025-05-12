import numpy as np
from functions import myGAT, similarity_measure
from scipy.special import comb

n = 3000
n1 = n2 = round(n/2)
gt1 = np.ones((n1))
gt2 = np.ones((n2)) * (-1)
gt = np.concatenate((gt1, gt2))

sigma = 10
mu = 2 * sigma * np.sqrt(np.log(n))

p_in = 3 * np.square(np.log(n)) / n  # Intra-community connection probability
p_out = 2 * np.square(np.log(n)) / n  # Inter-community connection probability

t = 0  # t = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
L = 100

T = np.ones((L)) * t

"""
Generate adjacency matrix A
"""
# Create the adjacency matrix for the stochastic block model
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

np.save('A1.npy', A)

# """
# Load adjacency matrix A
# """
# A = np.load('A1.npy')

"""
Generate feature matrix X
"""
X1 = np.random.normal(mu, sigma, n1)
X2 = np.random.normal(-1 * mu, sigma, n2)

X = np.concatenate((X1, X2))
X = np.reshape(X, (n, 1))

np.save('X1.npy', X)

# """
# Load feature matrix X
# """
# X = np.load('X1.npy')

"""
Use graph neural network with attention mechanism
"""

def calculate_accuracy(vector_a, vector_b):
    """
    Calculate the ratio of differing elements between two vectors.

    Parameters:
    vector_a: The first vector, values should be either 1 or -1.
    vector_b: The second vector, values should be either 1 or -1.

    Returns:
    differences_ratio: The proportion of differing elements.
    """
    # Calculate positions where the corresponding elements in both vectors are different
    differences = vector_a != vector_b

    # Count the number of differing elements
    num_differences = np.sum(differences)

    # Calculate the proportion of differing elements
    total_elements = len(vector_a)
    mis_ratio = num_differences / total_elements
    acc = 1 - mis_ratio

    return acc

X_temp = X
print(np.mean(X[:n1, :], axis=0), np.var(X[:n1, :], axis=0))

label = np.sign(X)
label = np.reshape(label, (n))
acc_ori = calculate_accuracy(gt, label)
print("acc_ori", acc_ori)

simi_data = np.empty(0, dtype=int)
for t in T:
    similarity = similarity_measure(X_temp)
    print("similarity", similarity)
    simi_data = np.append(simi_data, similarity)
    X_out = myGAT(A, X_temp, n1, n2, t)
    X_temp = X_out
    label = np.sign(X_out)
    label = np.reshape(label, (n))
    acc = calculate_accuracy(gt, label)
    print("acc:", acc)
print(simi_data)

np.savetxt("./oversmoothing_data/simi_data_{}.npy".format(t), simi_data)

# loaded_matrix = np.loadtxt("./oversmoothing_data/simi_data_0.0.npy")
