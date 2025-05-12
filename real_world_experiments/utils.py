import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from torch_geometric.datasets import Planetoid

def plot_multiple_means_with_confidence_intervals(datasetname, t_values, data_list, labels=None):
    """
    Plot the means and 95% confidence intervals of multiple data matrices.

    Args:
    t_values (numpy.ndarray): Vector of t values.
    data_list (list of numpy.ndarray): List of n*m data matrices.
    labels (list of str, optional): Labels for each data matrix. If provided, they will be displayed in the legend.
    """
    if labels is None:
        labels = [f'Dataset {i + 1}' for i in range(len(data_list))]

    plt.figure(figsize=(6, 5))

    for data, label in zip(data_list, labels):
        # Ensure t_values and data dimensions match
        if len(t_values) != data.shape[0]:
            raise ValueError("The length of t_values must be equal to the number of rows in the data")

        n, m = data.shape

        # Calculate means and standard errors
        means = np.mean(data, axis=1)
        std_errors = np.std(data, axis=1, ddof=1) / np.sqrt(m)  # Standard error

        # Calculate 95% confidence interval
        confidence_interval = 1.96 * std_errors  # 1.96 is the Z-score for 95% confidence

        # Plotting
        color = plt.cm.tab10(len(plt.gca().lines))  # Generate different colors

        plt.plot(t_values, means, '-o', color=color, label=label)
        plt.fill_between(t_values, means - confidence_interval, means + confidence_interval, color=color, alpha=0.2)

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xscale('log')
    plt.xlabel(r'Distance between means', fontsize=20)
    plt.ylabel('Classification Accuracy', fontsize=20)
    plt.title(datasetname, fontsize=25)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def add_noise_to_node_features(features, noise_level):
    """
    Add noise to node features.

    :param features: Original node features, shape (num_nodes, num_features)
    :param noise_level: Noise intensity, standard deviation
    :return: Node features with added noise
    """
    # Generate noise with the same shape as the feature matrix
    # Ensure the target device
    device = features.device  # Get the device of features
    # Move noise to the same device
    noise = torch.normal(mean=0, std=noise_level, size=features.size())
    noise = noise.to(device)
    # Add noise to the original features
    noisy_features = features + noise
    return noisy_features

def add_gaussian_noise_to_columns(data_matrix, t):
    """
    Add Gaussian noise with mean 0 and variance t to each column (feature) of the data matrix.

    Args:
    data_matrix (torch.Tensor): Input data matrix, where each column represents a feature.
    t (float): Variance of Gaussian noise.

    Returns:
    torch.Tensor: Data matrix with added noise.
    """
    # Generate a noise matrix with the same shape as data_matrix, mean 0, and std sqrt(t)
    device = data_matrix.device
    noise = torch.normal(mean=0.0, std=torch.sqrt(torch.tensor(t)), size=data_matrix.shape).to(device)

    # Add noise to each column
    noisy_data_matrix = data_matrix + noise

    return noisy_data_matrix

def add_noise_to_adjacency_matrix(adj_matrix, noise_strength=0):
    """
    Add noise to the adjacency matrix.

    Args:
    adj_matrix (torch.Tensor): Original adjacency matrix (square matrix).
    noise_strength (float): Noise intensity, a probability between 0 and 1 representing the chance of adding or removing edges.

    Returns:
    torch.Tensor: Adjacency matrix with added noise.
    """
    # Check if the adjacency matrix is square
    assert adj_matrix.size(0) == adj_matrix.size(1), "Adjacency matrix must be square"

    # Create a random matrix of the same size as the adjacency matrix
    noise = torch.rand_like(adj_matrix, dtype=torch.float32)

    # Create a mask matrix to indicate which positions to add or remove edges
    mask = (noise < noise_strength)

    # Add or remove edges
    noisy_adj_matrix = adj_matrix.clone()
    noisy_adj_matrix[mask] = 1 - noisy_adj_matrix[mask]

    # Ensure symmetry
    noisy_adj_matrix = torch.triu(noisy_adj_matrix, diagonal=1)
    noisy_adj_matrix = noisy_adj_matrix + noisy_adj_matrix.T

    return noisy_adj_matrix

def add_balanced_noise_to_adjacency_matrix(adj_matrix, noise_strength=0.1):
    """
    Add noise to the adjacency matrix while keeping the total number of edges unchanged.

    Args:
    adj_matrix (torch.Tensor): Original adjacency matrix (square matrix).
    noise_strength (float): Noise intensity, representing the proportion of edges to add or remove.

    Returns:
    torch.Tensor: Adjacency matrix with added noise.
    """
    # Ensure the adjacency matrix is square
    assert adj_matrix.size(0) == adj_matrix.size(1), "Adjacency matrix must be square"

    num_nodes = adj_matrix.size(0)

    # Get the current number of edges (excluding self-loops)
    current_edges = adj_matrix.nonzero(as_tuple=False)
    num_edges = current_edges.size(0) // 2  # For undirected graphs, the number of edges is halved

    # Calculate the number of edges to add or remove
    num_noisy_edges = int(noise_strength * num_edges)

    # Randomly select edges to remove
    delete_indices = torch.randperm(num_edges)[:num_noisy_edges]
    delete_edges = current_edges[delete_indices]

    # Create a mask to remove edges
    noisy_adj_matrix = adj_matrix.clone()
    noisy_adj_matrix[delete_edges[:, 0], delete_edges[:, 1]] = 0
    noisy_adj_matrix[delete_edges[:, 1], delete_edges[:, 0]] = 0

    # Get the indices of non-edges
    non_edges = (noisy_adj_matrix == 0).nonzero(as_tuple=False)

    # Randomly select edges to add
    add_indices = torch.randperm(non_edges.size(0))[:num_noisy_edges]
    add_edges = non_edges[add_indices]

    # Add edges
    noisy_adj_matrix[add_edges[:, 0], add_edges[:, 1]] = 1
    noisy_adj_matrix[add_edges[:, 1], add_edges[:, 0]] = 1

    return noisy_adj_matrix

def add_class_based_noise(adj_matrix, labels, same_class_prob=0.1, diff_class_prob=0.1):
    """
    Add noise based on node class: for nodes of the same class, randomly delete existing edges; for nodes of different classes, randomly add edges.

    Args:
    adj_matrix (torch.Tensor): Adjacency matrix (square matrix).
    labels (torch.Tensor): Labels for each node, shape [num_nodes].
    same_class_prob (float): Probability of deleting edges between nodes of the same class.
    diff_class_prob (float): Probability of adding edges between nodes of different classes.

    Returns:
    torch.Tensor: Adjacency matrix with added noise.
    """
    num_nodes = adj_matrix.size(0)
    noisy_adj_matrix = adj_matrix.clone()

    # Create masks for nodes of the same class and different classes
    same_class_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    diff_class_mask = 1 - same_class_mask

    # Get existing edges between nodes of the same class
    same_class_edges = (noisy_adj_matrix * same_class_mask).nonzero(as_tuple=False)

    # Randomly select edges to delete between nodes of the same class
    num_same_class_edges = same_class_edges.size(0)
    delete_count = int(same_class_prob * num_same_class_edges)
    if delete_count > 0:
        delete_indices = torch.randperm(num_same_class_edges)[:delete_count]
        delete_edges = same_class_edges[delete_indices]
        noisy_adj_matrix[delete_edges[:, 0], delete_edges[:, 1]] = 0
        noisy_adj_matrix[delete_edges[:, 1], delete_edges[:, 0]] = 0  # Ensure symmetry

    # Get non-edges between nodes of different classes
    diff_class_non_edges = ((1 - noisy_adj_matrix) * diff_class_mask).nonzero(as_tuple=False)

    # Randomly select edges to add between nodes of different classes
    num_diff_class_non_edges = diff_class_non_edges.size(0)
    add_count = int(diff_class_prob * num_diff_class_non_edges)
    if add_count > 0:
        add_indices = torch.randperm(num_diff_class_non_edges)[:add_count]
        add_edges = diff_class_non_edges[add_indices]
        noisy_adj_matrix[add_edges[:, 0], add_edges[:, 1]] = 1
        noisy_adj_matrix[add_edges[:, 1], add_edges[:, 0]] = 1  # Ensure symmetry

    return noisy_adj_matrix

def count_intra_inter_edges(adj_matrix, labels):
    """
    Count the number of intra-class and inter-class edges.

    Args:
    adj_matrix (torch.Tensor): Adjacency matrix (square matrix).
    labels (torch.Tensor): Labels for each node, shape [num_nodes].

    Returns:
    Tuple: Number of intra-class edges, number of inter-class edges.
    """
    num_nodes = adj_matrix.size(0)

    # Create masks for nodes of the same class and different classes
    same_class_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    diff_class_mask = 1 - same_class_mask

    # Count intra-class edges
    intra_class_edges = (adj_matrix * same_class_mask).sum().item() / 2

    # Count inter-class edges
    inter_class_edges = (adj_matrix * diff_class_mask).sum().item() / 2

    return intra_class_edges, inter_class_edges


def create_train_mask(gt, train_ratio=0.2):
    """
    Generate a training mask, ensuring that train_ratio of samples is selected from each class.

    Parameters:
    gt (Tensor): Node labels, a 1D tensor of size (N,), where N is the number of nodes.
    train_ratio (float): The proportion of samples to select from each class.

    Returns:
    Tensor: A training mask, a boolean tensor of the same size as gt.
    """
    num_nodes = gt.size(0)
    unique_labels = gt.unique()  # Get all unique labels
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)  # Initialize a mask of all False values

    for label in unique_labels:
        # Get the indices of all samples with the current label
        indices = torch.nonzero(gt == label).squeeze()

        # Calculate the number of samples to select
        num_train_samples = int(len(indices) * train_ratio)

        # Randomly select samples
        train_indices = indices[torch.randperm(len(indices))[:num_train_samples]]

        # Update the training mask
        train_mask[train_indices] = True

    return train_mask

def subtract_class_mean(x, gt, mu):
    # For data with label 0
    class_0_mask = (gt == 0)  # Get the mask for label 0
    class_0_data = x[class_0_mask]  # Extract the data with label 0
    class_0_mean = class_0_data.mean(dim=0, keepdim=True)  # Compute the mean of the label 0 data
    x[class_0_mask] = class_0_data - class_0_mean + mu  # Subtract class mean from the data for each class

    # For data with label 1
    class_1_mask = (gt == 1)  # Get the mask for label 1
    class_1_data = x[class_1_mask]  # Extract the data with label 1
    class_1_mean = class_1_data.mean(dim=0, keepdim=True)  # Compute the mean of the label 1 data
    x[class_1_mask] = class_1_data - class_1_mean - mu  # Subtract class mean from the data for each class

    return x
