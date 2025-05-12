from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from utils import plot_multiple_means_with_confidence_intervals, add_gaussian_noise_to_columns
from submit_code.real_world_experiments.models import GCN, GAT, GCN_GAT
import numpy as np
# training process
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# prepare dataset
dataset = Planetoid(root='./tmp/Cora', name='Cora')
dataset_name = 'Cora'
data = dataset[0]

mean = data.x.mean(dim=0).mean(dim=0).item()
noise_level = np.arange(0, 0.07, 0.07/10)
# noise_level_y_label = np.arange(0, 0.07, 0.07/10)
trials = 100

acc1_matrix = np.zeros((noise_level.shape[0], trials))
acc2_matrix = np.zeros((noise_level.shape[0], trials))
acc3_matrix = np.zeros((noise_level.shape[0], trials))

for j in range(noise_level.shape[0]):
    value = noise_level[j]
    data.x = add_gaussian_noise_to_columns(data.x, value).to(device)
    data.to(device)

    acc1 = []
    acc2 = []
    acc3 = []
    for i in range(trials):
        # 构建模型
        # GCN
        # model1 = GCN4(data.x.shape[1], 32, dataset.num_classes).to(device)
        model1 = GCN(in_channels=data.x.shape[1], out_channels=dataset.num_classes).to(device)
        # GAT
        # model2 = GAT4(in_channels=data.x.shape[1], hidden_channels=16, out_channels=dataset.num_classes, heads=2).to(device)
        model2 = GAT(in_channels=data.x.shape[1], out_channels=dataset.num_classes).to(device)

        model3 = GCN_GAT(data.x.shape[1], dataset.num_classes).to(device)


        optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001, weight_decay=5e-4)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001, weight_decay=5e-4)
        optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001, weight_decay=5e-4)
        epochs = 100

        model1.train()
        for epoch in range(epochs):
            optimizer1.zero_grad()
            out1 = model1(data)
            # loss1 = F.nll_loss(out1[data.train_mask], data.y[data.train_mask])
            loss1 = F.cross_entropy(out1[data.train_mask], data.y[data.train_mask])
            # print("loss 1:", loss1)
            loss1.backward()
            optimizer1.step()

        model2.train()
        for epoch in range(epochs):
            optimizer2.zero_grad()
            out2 = model2(data)
            # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss2 = F.cross_entropy(out2[data.train_mask], data.y[data.train_mask])
            # print("loss 2:", loss2)
            loss2.backward()
            optimizer2.step()

        model3.train()
        for epoch in range(epochs):
            optimizer3.zero_grad()
            out3 = model3(data)
            # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss3 = F.cross_entropy(out3[data.train_mask], data.y[data.train_mask])
            # print("loss 3:", loss3)
            loss3.backward()
            optimizer3.step()



        # testing process
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
        # print(f'Accuracy for GCNGAT: {acc3:.4f}')

    acc1 = np.array(acc1)
    acc2 = np.array(acc2)
    acc3 = np.array(acc3)

    acc1_matrix[j, :] = acc1
    acc2_matrix[j, :] = acc2
    acc3_matrix[j, :] = acc3

# print(acc1_matrix)
# print(acc2_matrix)
# print(acc3_matrix)
acc1_matrix = np.array(acc1_matrix)
acc2_matrix = np.array(acc2_matrix)
acc3_matrix = np.array(acc3_matrix)
np.savetxt("./experimental_data/acc1_matrix_{}.txt".format(dataset_name), acc1_matrix)
np.savetxt("./experimental_data/acc2_matrix_{}.txt".format(dataset_name), acc2_matrix)
np.savetxt("./experimental_data/acc3_matrix_{}.txt".format(dataset_name), acc3_matrix)


data_list = []
data_list.extend([acc1_matrix, acc2_matrix, acc3_matrix])

label = ["GCN", "GAT", "GAT*"]

plot_multiple_means_with_confidence_intervals(dataset_name, noise_level, data_list, labels=label)




# Calculate the total number of trainable parameters
# total_params1 = sum(param.numel() for param in model1.parameters() if param.requires_grad)
# total_params2 = sum(param.numel() for param in model2.parameters() if param.requires_grad)
# total_params3 = sum(param.numel() for param in model3.parameters() if param.requires_grad)
#
# print(f"Total number of trainable parameters: {total_params1, total_params2, total_params3}")