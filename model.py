import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
import numpy as np

# Encoder architecture
class Encoder(nn.Module):
    def __init__(self, in_channel, fea_dim, cluster_num):
        super(Encoder, self).__init__()
        self.fea_dim = fea_dim
        self.in_channel = in_channel
        self.cluster_num = cluster_num
        self.net = nn.Sequential(
            nn.Linear(self.in_channel, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, self.fea_dim * 2),
            nn.BatchNorm1d(self.fea_dim * 2),
            nn.ReLU(True),
        )
        self.predict = nn.Sequential(
            nn.Linear(self.fea_dim * 2, self.fea_dim),
            nn.BatchNorm1d(self.fea_dim),
            nn.ReLU(True))
        self.clustering = nn.Sequential(
            nn.Linear(self.fea_dim, self.cluster_num),
        )

    def forward(self, x):
        params = self.net(x)
        mu, sigma = params[:, :self.fea_dim], params[:, self.fea_dim:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive
        p_z1_given_v1 = Independent(Normal(loc=mu, scale=sigma), 1)
        return p_z1_given_v1


class Encoder_f(nn.Module):
    def __init__(self, view_num, in_channel, fea_dim, cluster_num):
        super(Encoder_f, self).__init__()
        self.view_num = view_num
        self.fea_dim = fea_dim
        self.cluster_num = cluster_num
        self.all_encoder = nn.ModuleList([Encoder(in_channel[i], fea_dim, cluster_num).cuda() for i in range(view_num)])
        self.net1 = nn.Sequential(
            nn.Linear(self.fea_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, self.fea_dim * 2),
            nn.BatchNorm1d(self.fea_dim * 2),
            nn.ReLU(True),
        )
        self.predict1 = nn.Sequential(
            nn.Linear(self.fea_dim * 2, self.fea_dim),
            nn.BatchNorm1d(self.fea_dim),
            nn.ReLU(True)
        )
        self.cluster1 = nn.Sequential(
            nn.Linear(self.fea_dim, self.cluster_num),
        )
        self.weights = nn.Parameter(torch.full((view_num,), 1 / view_num), requires_grad=True)

    def forward(self, input):
        x_P = [self.all_encoder[i](input[i]) for i in range(self.view_num)]
        x_z = [p.rsample() for p in x_P]

        weights = nn.functional.softmax(self.weights, dim=0)

        x_fusion = torch.sum(weights[None, None, :] * torch.stack(x_z, dim=-1), dim=-1)
        params = self.net1(x_fusion)
        mu, sigma = params[:, :self.fea_dim], params[:, self.fea_dim:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive
        x_P_F = Independent(Normal(loc=mu, scale=sigma), 1)
        # x_P_F_z = x_P_F.rsample()
        # clustering_f = self.cluster1(x_P_F_z)

        x_F = [self.predict1(self.net1(x_z[i])) for i in range(self.view_num)]
        clustering = [self.all_encoder[i].clustering(x_F[i]) for i in range(self.view_num)]
        # clustering = [self.cluster1(x_F[i]) for i in range(self.view_num)]
        clustering = [torch.softmax(clustering1, dim=1) for clustering1 in clustering]
        # clustering.append(clustering_f)
        return x_P_F, x_P, clustering


# Auxiliary network for mutual information estimation
class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(MIEstimator, self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
        )

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        temp = torch.roll(x1, 1, 0)
        neg = self.net(torch.cat([temp, x2], 1))
        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1


def UD_constraint(classer):
    CL = classer.detach().cpu().numpy()
    N, K = CL.shape
    CL = CL.T
    r = np.ones((K, 1)) / K
    c = np.ones((N, 1)) / N
    CL **= 10
    inv_K = 1. / K
    inv_N = 1. / N
    err = 1e3
    _counter = 0
    while err > 1e-2 and _counter < 100:
        r = inv_K / (CL @ c)
        c_new = inv_N / (r.T @ CL).T
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1
    CL *= np.squeeze(c)
    CL = CL.T
    CL *= np.squeeze(r)
    CL = CL.T
    argmaxes = np.nanargmax(CL, 0)
    newL = torch.LongTensor(argmaxes)
    return newL


