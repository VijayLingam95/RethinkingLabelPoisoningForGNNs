import torch
import torch_geometric as tg
import torch.nn.functional as F

from torch_geometric.datasets import CitationFull
from torch_geometric.nn import GCNConv

import numpy as np
import scipy.sparse as sp
import scipy.linalg as spl


import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import seaborn as sns


#seeding
def reset_seed(SEED=0):
	torch.manual_seed(SEED)
	np.random.seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)




def propagation_matrix(adj, alpha=0.85, sigma=1, nodes=None):
    """
    Computes the propagation matrix  (1-alpha)(I - alpha D^{-sigma} A D^{sigma-1})^{-1}.
    Parameters
    ----------
    adj : sp.spmatrix, shape [n, n]
        Sparse adjacency matrix.
    alpha : float
        (1-alpha) is the teleport probability.
    sigma
        Hyper-parameter controlling the propagation style.
        Set sigma=1 to obtain the PPR matrix.
    nodes : np.ndarray, shape [?]
        Nodes for which we want to compute Personalized PageRank.
    Returns
    -------
    prop_matrix : np.ndarray, shape [n, n]
        Propagation matrix.
    """
    n = adj.shape[0]
    deg = adj.sum(1).A1

    deg_min_sig = sp.diags(np.power(deg, -sigma))
    deg_sig_min = sp.diags(np.power(deg, sigma - 1))
    pre_inv = sp.eye(n) - alpha * deg_min_sig @ adj @ deg_sig_min

    # solve for x in: pre_inx @ x = b
    b = np.eye(n)
    if nodes is not None:
        b = b[:, nodes]

    return (1 - alpha) * spl.solve(pre_inv.toarray().T, b).T


def get_margin_gcn(model, dataset, train_mask, device, n_epoch=20 , y_poison=None):
    data= dataset.data
    eye = torch.eye(dataset.num_classes).to(device)
    h = torch.ones((data.num_nodes, dataset.num_classes)) / dataset.num_classes
    h=h.to(device)
    h[train_mask] = eye[data.y[train_mask]]

    sel = torch.ones_like(h).type(torch.bool).to(device)
    sel[torch.arange(data.num_nodes), data.y] = 0

    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(n_epoch):
        optimizer.zero_grad()
        out = model(data)
        if y_poison is None:
            loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        else:
            loss = F.cross_entropy(out[data.train_mask], y_poison)
        loss.backward()  # Derive gradients.
        optimizer.step()  # U

    model.eval()
    pred = model(data)

    p_true = pred[np.arange(data.num_nodes), data.y]
    p_second = torch.reshape(pred[sel], (data.num_nodes, dataset.num_classes-1)).max(1).values

    margin = p_true - p_second
    return margin.detach()

def get_margin_lp(train_mask, y, adj, pi, dataset, device):
    data = dataset.data
    eye = torch.eye(dataset.num_classes).to(device)
    h = torch.ones((data.num_nodes, dataset.num_classes)) / dataset.num_classes
    h=h.to(device)
    h[train_mask] = eye[y]
    tie = (torch.arange(1, dataset.num_classes+1)/10000)[None, :]
    tie= tie.to(device)
  
    sel = torch.ones_like(h).type(torch.bool)
    sel[torch.arange(data.num_nodes), data.y] = 0


    pred = pi @ h + tie

    p_true = pred[np.arange(data.num_nodes), data.y]
    p_second = torch.reshape(pred[sel], (data.num_nodes, dataset.num_classes-1)).max(1).values

    margin = p_true - p_second

    return margin.detach()

def margin_attack_labels(model, dataset, args, BUDGET, device, verbose=False):

    reset_seed()
    
    data = dataset.data
    num_classes = data.y.max() + 1
    eye = torch.eye(num_classes).to(device)
    tie = (np.arange(1, dataset.num_classes+1)/10000)[None, :]

    h = torch.ones((data.num_nodes, dataset.num_classes)) / dataset.num_classes
    h = h.to(device)
    h[data.train_mask] = eye[data.y[data.train_mask]]
  
    sel = torch.ones_like(h).type(torch.bool)
    sel[torch.arange(data.num_nodes), data.y] = 0



    # np.random.seed(0)
    # torch.manual_seed(0)
    # torch.manual_cuda()

    n_samples = 500
    n_train_lr = int(0.9*n_samples)
    n_train = data.train_mask.sum()
    p = torch.ones(n_train)*0.5

    subsets = torch.zeros((n_samples, n_train)).to(device)

    margins_lp = torch.zeros((n_samples, data.num_nodes)).to(device)
    test_accs_lp = torch.zeros(n_samples).to(device)

    margins_gcn = torch.zeros((n_samples, data.num_nodes)).to(device)
    test_accs_gcn = torch.zeros(n_samples).to(device)

    adj = tg.utils.to_scipy_sparse_matrix(data.edge_index)
    pi = propagation_matrix(adj)
    pi = torch.Tensor(pi).to(device)

    for i in tqdm(range(n_samples)):
        train_mask = data.train_mask.clone()
        subset = torch.bernoulli(p).to(device)
        train_mask[train_mask == True] = subset.type(torch.bool)

        subsets[i] = subset

        margin = get_margin_lp(train_mask, data.y[train_mask],adj, pi, dataset, device)
        margins_lp[i] = margin
        test_accs_lp[i] = (margin > 0)[data.test_mask].sum() / data.test_mask.sum()


    from sklearn.ensemble import RandomForestRegressor
    # lr = RandomForestRegressor()
    lr = LinearRegression()
    lr_margin = LinearRegression()
    lr_margin.fit(subsets.cpu().numpy(), margins_lp.cpu().numpy())
    lr.fit(subsets.cpu().numpy(), test_accs_lp.cpu().numpy())

    rnd = (torch.rand(sel.shape).to(device)+0.000000001) * sel
    rnd = rnd[data.train_mask]
    to_flip = torch.tensor(lr.coef_.argsort()[-BUDGET:]).to(device)
    # to_flip = torch.tensor(lr.feature_importances_.argsort()[:-BUDGET:]).to(device)
    y_poison = data.y[data.train_mask].clone()
    y_poison[to_flip] = rnd[to_flip].argmax(1)

    return y_poison