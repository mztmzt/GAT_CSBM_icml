import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import GATConv, GCNConv

# implement a two-layer GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, concat=True, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class GCN_GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN_GAT, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        # self.conv1 = GATConv(in_channels, 8, heads=8, concat=True, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class SingleHeadGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, t=1.0):
        super(SingleHeadGATConv, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)  # Linear transformation
        self.bias = Parameter(torch.Tensor(out_channels))  # Optional bias term
        self.t = t  # Custom attention parameter
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linearly transform node feature matrix
        x = self.lin(x)

        # Start message passing
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index, size_i):
        # Compute attention coefficients
        alpha = torch.sum(x_i * x_j, dim=-1)  # Compute dot product

        # Apply custom attention score logic
        alpha = torch.where(alpha > 0, torch.tensor(self.t, dtype=x_i.dtype, device=x_i.device),
                            torch.tensor(-self.t, dtype=x_i.dtype, device=x_i.device))

        # Normalize attention coefficients
        alpha = softmax(alpha, edge_index[0], num_nodes=size_i)

        # Return the weighted message
        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        # Optionally add bias
        if self.bias is not None:
            aggr_out += self.bias
        return aggr_out


class SimpleGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SimpleGATConv, self).__init__(aggr='add')  # Use add aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        # self.att = Parameter(torch.Tensor(1, out_channels * 2))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        # torch.nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.propagate(edge_index, x=x)
        x = self.lin(x)

        return x

    def message(self, x_i, x_j, edge_index, size_i):
        # Compute attention coefficients
        vec = torch.ones((x_i.shape[1], 1)) / (x_i.shape[1] ** (0.5))
        x_i_v = x_i @ vec.to(device=x_i.device)
        x_j_v = x_j @ vec.to(device=x_i.device)
        x_cat = torch.cat([x_i_v, x_j_v], dim=-1)  # Concatenate features
        atten_layer_1 = torch.tensor([[1.0, -1.0, 1.0, -1.0], [1.0, -1.0, -1.0, 1.0]]).to(device=x_i.device)
        atten_layer_2 = torch.tensor([1.0, 1.0, -1.0, -1.0]).to(device=x_i.device)
        alpha = F.leaky_relu(x_cat @ atten_layer_1, negative_slope=0.1)
        alpha = alpha @ atten_layer_2 * 5
        # alpha = (x_cat * self.att).sum(dim=-1)  # Compute attention scores
        # alpha = F.leaky_relu(alpha, negative_slope=0.2)  # Use LeakyReLU
        alpha = softmax(alpha, edge_index[0], num_nodes=size_i)  # Normalize attention coefficients
        out = x_j * alpha.unsqueeze(-1)

        # Return weighted neighbor features
        return out

    def update(self, aggr_out):
        # Return aggregated node features
        return aggr_out

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class GCN1(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        # self.conv2 = GCNConv(64, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class GAT_my_attention(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SingleHeadGATConv(in_channels, 64, t=1)
        self.conv2 = SingleHeadGATConv(64, out_channels, t=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class GAT_my_attention1(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SingleHeadGATConv(in_channels, out_channels, t=1)
        # self.conv2 = SingleHeadGATConv(64, out_channels, t=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class GAT_jmlr(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SimpleGATConv(in_channels, 64)
        self.conv2 = SimpleGATConv(64, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class GAT_jmlr1(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SimpleGATConv(in_channels, out_channels)
        # self.conv2 = SimpleGATConv(64, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
