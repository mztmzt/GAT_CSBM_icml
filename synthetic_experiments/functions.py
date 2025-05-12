import numpy as np

def myGAT(A1, Xx, n1, n2, t):

    X_A = Xx @ Xx.T
    X_A = np.where(X_A > 0, t, np.where(X_A < 0, -t, X_A))
    X_A = X_A * A1

    # Calculate the exponent of each element
    Exp = np.exp(X_A) * A1

    # Row-wise sum of exponents
    Sums = np.sum(Exp, axis=1, keepdims=True)

    # Calculate softmax
    Softmax_X_A = Exp / Sums
    # print(Softmax_X_A)

    X_2 = Softmax_X_A @ Xx

    X_2_c1 = X_2[:n1, :]
    X_2_c2 = X_2[n1:, :]

    # mean_c1 = np.mean(X_2_c1, axis=0)
    # var_c1 = np.var(X_2_c1, axis=0)
    # mean_c2 = np.mean(X_2_c2, axis=0)
    # var_c2 = np.var(X_2_c2, axis=0)
    # print("mean:", mean_c1, mean_c2, "var:", var_c1, var_c2)

    return X_2

def similarity_measure(xout):
    a = np.copy(xout)
    mout = np.mean(a)
    for i in range(len(a)):
        a[i] = (a[i] - mout)**2
    sum1 = np.sum(a) / len(a)
    sum1 = np.sqrt(sum1)
    return sum1

def calculate_accuracy(vector_a, vector_b):
    """
    Calculate the proportion of differing elements between two vectors.

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
