import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, EdgeConv, GATConv
import torch

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.act_fn(x)
        x = self.layer2(x)

        return x

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_dim, hid_dim))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim))
            self.convs.append(GraphConv(hid_dim, hid_dim))

        self.layer1 = nn.Linear(hid_dim, out_dim, bias=True)

    def forward(self, graph, x):

        for i in range(self.n_layers - 1):
            x = F.elu(self.convs[i](graph, x))
        x = self.convs[-1](graph, x)
        x = self.layer1(F.elu(x))
        return x

class GAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GATConv(in_dim, hid_dim, num_heads = 3))


        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GATConv(hid_dim, hid_dim, num_heads = 3))
            self.convs.append(GATConv(hid_dim, hid_dim, num_heads = 3))

        self.layer1 = nn.Linear(hid_dim, out_dim, bias=True)

    def forward(self, graph, x):

        for i in range(self.n_layers - 1):
            x = F.elu(torch.mean(self.convs[i](graph, x), dim=1))
        x = torch.mean(self.convs[-1](graph, x), dim=1)
        x = self.layer1(F.elu(x))
        return x

class GATE_GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, hid_dim,  weight = 'true'))


        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim))
            self.convs.append(GraphConv(hid_dim, out_dim))

        self.layer0 = nn.Linear(in_dim, hid_dim)
        self.bn0 = nn.BatchNorm1d(hid_dim, affine=False)
        self.layer1 = nn.Linear(out_dim, out_dim)

    def forward(self, graph, x):
        for i in range(self.n_layers - 1):
            x = F.elu(self.convs[i](graph, x))
        x = self.convs[-1](graph, x)
        x = self.layer1(F.elu(x))
        return x

class GATE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, out_dim2, n_layers):
        super().__init__()
        self.backbone = GATE_GCN(in_dim, hid_dim, out_dim, n_layers)
        self.cls = MLP_Predictor(out_dim, 2, out_dim2)

    def get_embedding(self, graph, feat):
        out = self.backbone(graph, feat)
        return out.detach()

    def finetune(self, graph, feat):
        out = self.backbone(graph, feat)
        out = F.elu(out)
        c = self.cls(out)
        return c

    def forward(self, A_a, X_a, A_b, X_b):
        X_a = F.dropout(X_a, 0.2)
        X_b = F.dropout(X_b, 0.2)

        embeding_a = self.backbone(A_a, X_a)
        embeding_b = self.backbone(A_b, X_b)

        embeding_a = (embeding_a - embeding_a.mean(0)) / embeding_a.std(0)
        embeding_b = (embeding_b - embeding_b.mean(0)) / embeding_b.std(0)

        c_a = self.cls(embeding_a)
        c_b = self.cls(embeding_b)
        return embeding_a, embeding_b, c_a, c_b

class MLP_Predictor(nn.Module):
    r"""MLP used for predictor. The MLP has one hidden layer.

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()


        self.net = nn.Sequential(
            nn.Linear(input_size, output_size, bias=True)
        )

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

