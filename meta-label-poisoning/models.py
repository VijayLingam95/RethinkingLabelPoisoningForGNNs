import math
from copy import deepcopy
import numpy as np
import scipy.sparse as sp
import utils_extra as utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch.optim as optim

from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.utils import from_scipy_sparse_matrix

# seed
torch.manual_seed(0)


class Linear1(torch.nn.Module): 
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear1, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN_MG(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_MG, self).__init__()
        self.Linear1 = Linear1(nfeat, nhid, dropout, bias=True)
        self.Linear2 = Linear1(nhid, nclass, dropout, bias=True)

    def forward(self, data):
        x, adj = data.x, data.adj
        x = torch.relu(self.Linear1(torch.matmul(adj, x)))
        h = self.Linear2(torch.matmul(adj, x))
        return torch.log_softmax(h, dim=-1)


class GCN(torch.nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class MLP(torch.nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x


class GAT(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class APPNP_net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return x


class GCN_JKNet(torch.nn.Module):
    def __init__(self, dataset, args):
        in_channels = dataset.num_features
        out_channels = dataset.num_classes

        super(GCN_JKNet, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin1 = torch.nn.Linear(16, out_channels)
        self.one_step = APPNP(K=1, alpha=0)
        self.JK = JumpingKnowledge(mode='lstm',
                                   channels=16,
                                   num_layers=4
                                   )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x = self.JK([x1, x2])
        x = self.one_step(x, edge_index)
        x = self.lin1(x)
        return x 


class CPGCN(torch.nn.Module):

    def __init__(self, nfeat, nhid, nclass, ncluster, dropout=0.5, lr=0.01, weight_decay=5e-4 ,device=None):

        super(CPGCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass

        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.edge_index = None
        self.edge_weight = None
        self.features = None
        self.clusters = None

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.fc1 = nn.Linear(nhid, nclass)
        self.fc2 = nn.Linear(nhid, ncluster)
        

    def forward(self, x, edge_index, edge_weight):

        x = F.relu(self.gc1(x, edge_index,edge_weight))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index,edge_weight)

        pred = self.fc1(x)
        pred_cluster = self.fc2(x)
        return pred, pred_cluster

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def fit(self, features, adj, labels, clusters, idx_train, idx_val, lam=1, train_iters=200, verbose=False):
        """
        """
        self.initialize()

        self.edge_index, self.edge_weight = from_scipy_sparse_matrix(adj)
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.float().to(self.device)

        if sp.issparse(features):
            features = utils.sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        self.features = features.to(self.device)
        self.labels = torch.LongTensor(np.array(labels)).to(self.device)
        self.clusters = torch.LongTensor(np.array(clusters)).to(self.device)


        self._train_with_val(self.labels, self.clusters ,idx_train, idx_val, lam ,train_iters, verbose)


    def _train_with_val(self, labels, clusters, idx_train, idx_val, lam, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            pred, pred_cluster = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_pred = F.cross_entropy(pred[idx_train], labels[idx_train])
            loss_cluster = F.cross_entropy(pred_cluster[idx_train], clusters[idx_train])

            loss_train = loss_pred + lam * loss_cluster
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, pred loss: {:.4f}, cluster loss: {:.4f}'.format(i, loss_train, loss_cluster))

            self.eval()
            pred, pred_cluster = self.forward(self.features, self.edge_index, self.edge_weight)
            acc_val = utils.accuracy(pred[idx_val], labels[idx_val])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.pred = pred
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        pred, pred_cluster = self.forward(self.features, self.edge_index,self.edge_weight)
        loss_test = F.cross_entropy(pred[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(pred[idx_test], self.labels[idx_test])
        acc_cluster = utils.accuracy(pred_cluster[idx_test], self.clusters[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()),
              "cluster acc= {:.4f}".format(acc_cluster.item()))
        return pred