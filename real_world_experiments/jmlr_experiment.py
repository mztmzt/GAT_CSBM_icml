from torch_geometric.datasets import Planetoid
import torch
from models import GAT_my_attention, GCN, GAT_jmlr, GCN1, GAT_jmlr1, GAT_my_attention1
import torch.nn.functional as F
from utils import create_train_mask, subtract_class_mean, plot_multiple_means_with_confidence_intervals
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage

# Training process
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare dataset
dataset = Planetoid(root='/home/mazhongtian/GCN_GAT_nwpu/tmp/Citeseer', name='Citeseer')
dataset_name = 'Citeseer'
data = dataset[0].to(device)
# Assume ground_truth is a tensor containing labels
data.test_mask = (data.y == 0)
# print(int(data.test_mask.sum()))
# Assume ground_truth is a tensor containing labels
data.y[data.y != 0] = 1
# data.train_mask = create_train_mask(data.y)

# Construct a sequence of 20 uniformly spaced values from -4 to -1
mu_seq = np.linspace(-4, -1, 30)

# Take the exponential of 10 for each value
mu_seq = np.power(10, mu_seq)

trials = 10

acc1_matrix = np.zeros((mu_seq.shape[0], trials))
acc2_matrix = np.zeros((mu_seq.shape[0], trials))
acc3_matrix = np.zeros((mu_seq.shape[0], trials))

for j in range(mu_seq.shape[0]):
    mu = mu_seq[j]
    data.x = subtract_class_mean(data.x, data.y, mu)

    acc1 = []
    acc2 = []
    acc3 = []

    model1 = GCN1(in_channels=data.x.shape[1], out_channels=dataset.num_classes).to(device)
    # model2 = GAT_my_attention(in_channels=data.x.shape[1], out_channels=dataset.num_classes).to(device)
    model2 = GAT_jmlr1(in_channels=data.x.shape[1], out_channels=dataset.num_classes).to(device)

    model3 = GAT_my_attention1(in_channels=data.x.shape[1], out_channels=dataset.num_classes).to(device)

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01, weight_decay=0.0005)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01, weight_decay=0.0005)
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.01, weight_decay=0.0005)

    epochs = 100

    for i in range(trials):
        model1.train()
        model2.train()
        model3.train()
        for epoch in range(epochs):
            optimizer1.zero_grad()
            out1 = model1(data)
            # loss1 = F.nll_loss(out1[data.train_mask], data.y[data.train_mask])
            loss1 = F.cross_entropy(out1[data.train_mask], data.y[data.train_mask])
            # print("loss 1:", loss1.item())
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            out2 = model2(data)
            # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss2 = F.cross_entropy(out2[data.train_mask], data.y[data.train_mask])
            # print("loss 2:", loss2.item())
            loss2.backward()
            optimizer2.step()

            optimizer3.zero_grad()
            out3 = model3(data)
            # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss3 = F.cross_entropy(out3[data.train_mask], data.y[data.train_mask])
            # print("loss 2:", loss2.item())
            loss3.backward()
            optimizer3.step()

        # Testing process
        model1.eval()
        pred = model1(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc1_temp = int(correct) / int(data.test_mask.sum())
        acc1.append(acc1_temp)
        # print(f'Accuracy for GCN: {acc1:.4f}')

        model2.eval()
        pred = model2(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc2_temp = int(correct) / int(data.test_mask.sum())
        acc2.append(acc2_temp)
        # print(f'Accuracy for GAT: {acc2:.4f}')

        model3.eval()
        pred = model3(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc3_temp = int(correct) / int(data.test_mask.sum())
        acc3.append(acc3_temp)
        # print(f'Accuracy for GAT: {acc3:.4f}')

    acc1 = np.array(acc1)
    acc2 = np.array(acc2)
    acc3 = np.array(acc3)

    acc1_matrix[j, :] = acc1
    acc2_matrix[j, :] = acc2
    acc3_matrix[j, :] = acc3

acc1_matrix = np.array(acc1_matrix)
acc2_matrix = np.array(acc2_matrix)
acc3_matrix = np.array(acc3_matrix)
# np.savetxt("./experimental_data/acc1_matrix_{}_2.txt".format(dataset_name), acc1_matrix)
# np.savetxt("./experimental_data/acc2_matrix_{}_2.txt".format(dataset_name), acc2_matrix)
# np.savetxt("./experimental_data/acc3_matrix_{}_2.txt".format(dataset_name), acc3_matrix)

data_list = []
data_list.extend([acc1_matrix, acc2_matrix, acc3_matrix])

label = ["GCN", "GAT_jmlr23", "GAT_ours"]

plot_multiple_means_with_confidence_intervals(dataset_name, mu_seq, data_list, labels=label)
