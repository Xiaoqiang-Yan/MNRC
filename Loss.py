import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
import numpy as np
EPSILON = 1E-9
DEBUG_MODE = False


def UD_constraint(classer):
    # _, _, classer = model(input1, input2, input3, input4)
    CL = classer.detach().cpu().numpy()
    N, K = CL.shape
    CL = CL.T
    r = np.ones((K, 1)) / K
    c = np.ones((N, 1)) / N
    CL **= 5
    inv_K = 1. / K
    inv_N = 1. / N
    err = 1e3
    _counter = 0
    while err > 1e-2 and _counter < 75:
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


class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(MIEstimator, self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
        )

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        mi_gradient = -softplus(-pos).mean() - softplus(neg).mean()
        mi_estimation = pos.mean() - neg.exp().mean() + 1
        return - mi_gradient.mean()



import sys
EPS = sys.float_info.epsilon


def mutual_information(x_img, x_txt):
        _, k = x_img.size()
        p_i_j = compute_joint(x_img, x_txt)
        assert (p_i_j.size() == (k, k))
        temp1 = p_i_j.sum(dim=1).view(k, 1)
        p_i = temp1.expand(k, k).clone()
        temp2 = p_i_j.sum(dim=0).view(1, k)
        p_j = temp2.expand(k, k).clone()
        p_i_j[(p_i_j < EPS).data] = EPS
        p_j[(p_j < EPS).data] = EPS
        p_i[(p_i < EPS).data] = EPS
        loss = - p_i_j * (torch.log(p_i_j) - torch.log(p_j) - torch.log(p_i))
        loss = loss.sum()
        return loss


def compute_joint(x_img, x_txt):
        bn, k = x_img.size()
        assert (x_txt.size(0) == bn and x_txt.size(1) == k)
        p_i_j = x_img.unsqueeze(2) * x_txt.unsqueeze(1)
        p_i_j = p_i_j.sum(dim=0)
        p_i_j = (p_i_j + p_i_j.t()) / 2.
        p_i_j = p_i_j / p_i_j.sum()
        return p_i_j


def DDC1(predict, clustering, clustering_num):
    return d_cs(clustering, predict, clustering_num)


def DDC2(predict, clustering, clustering_num):
    n = clustering.size(0)
    return 2 / (n * (n - 1)) * triu(clustering @ torch.t(clustering))

def DDC3(predict, clustering, clustering_num):
    eye = torch.eye(clustering_num, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    m = torch.exp(-cdist(clustering, eye))
    return d_cs(m, predict, clustering_num)


def mask_type_transfer(mask):
    mask = mask.type(torch.bool)
    return mask


def get_pos_and_neg_mask(bs):
    ''' Org_NTXentLoss_mask '''
    zeros = torch.zeros((bs, bs), dtype=torch.uint8)
    eye = torch.eye(bs, dtype=torch.uint8)
    pos_mask = torch.cat([
        torch.cat([zeros, eye], dim=0), torch.cat([eye, zeros], dim=0),
    ], dim=1)
    neg_mask = (torch.ones(2*bs, 2*bs, dtype=torch.uint8) - torch.eye(
        2*bs, dtype=torch.uint8))
    pos_mask = mask_type_transfer(pos_mask)
    neg_mask = mask_type_transfer(neg_mask)
    return pos_mask, neg_mask


class NTXentLoss(nn.Module):
    """ NTXentLoss
    Args:
        tau: The temperature parameter.
    """

    def __init__(self, bs, tau=0.5, cos_sim=True, gpu=True, eps=1e-8):
        super(NTXentLoss, self).__init__()
        self.name = 'NTXentLoss_Org'
        self.tau = tau
        self.use_cos_sim = cos_sim
        self.gpu = gpu
        self.eps = eps

        if cos_sim:
            self.cosine_similarity = nn.CosineSimilarity(dim=-1)
            self.name += '_CosSim'

        # Get pos and neg mask

        print(self.name)

    def forward(self, zi, zj, target=None):
        '''
        input: {'zi': out_feature_1, 'zj': out_feature_2}
        target: one_hot lbl_prob_mat
        '''
        # zi, zj = F.normalize(input['zi'], dim=1), F.normalize(input['zj'], dim=1)
        bs = zi.shape[0]
        self.pos_mask, self.neg_mask = get_pos_and_neg_mask(bs)

        if self.gpu:
            self.pos_mask = self.pos_mask.cuda()
            self.neg_mask = self.neg_mask.cuda()

        z_all = torch.cat([zi, zj], dim=0)  # input1,input2: z_i,z_j
        # [2*bs, 2*bs] -  pairwise similarity
        if self.use_cos_sim:
            sim_mat = self.cosine_similarity(
                z_all.unsqueeze(1), z_all.unsqueeze(0)) / self.tau  # s_(i,j)
        else:
            sim_mat = torch.mm(z_all, z_all.t().contiguous()) / self.tau  # s_(i,j)

        sim_pos = torch.exp(sim_mat.masked_select(self.pos_mask).view(2*bs).clone())
        # [2*bs, 2*bs-1]
        sim_neg = torch.exp(sim_mat.masked_select(self.neg_mask).view(2*bs, -1).clone())

        # Compute loss
        loss = (- torch.log(sim_pos / (sim_neg.sum(dim=-1) + self.eps))).mean()

        return loss


def get_contrast_loss(name, **kwargs):
    if name == 'NTXentLoss':
        criterion = NTXentLoss

    return criterion(**kwargs)


def triu(X):
    # Sum of strictly upper triangular part
    return torch.sum(torch.triu(X, diagonal=1))


def _atleast_epsilon(X, eps=EPSILON):
    """
    Ensure that all elements are >= `eps`.

    :param X: Input elements
    :type X: th.Tensor
    :param eps: epsilon
    :type eps: float
    :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
    :rtype: th.Tensor
    """
    return torch.where(X < eps, X.new_tensor(eps), X)


def d_cs(A, K, n_clusters):
    """
    Cauchy-Schwarz divergence.

    :param A: Cluster assignment matrix
    :type A:  th.Tensor
    :param K: Kernel matrix
    :type K: th.Tensor
    :param n_clusters: Number of clusters
    :type n_clusters: int
    :return: CS-divergence
    :rtype: th.Tensor
    """
    nom = torch.t(A) @ K @ A
    dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)

    nom = _atleast_epsilon(nom)
    dnom_squared = _atleast_epsilon(dnom_squared, eps=EPSILON**2)

    d = 2 / (n_clusters * (n_clusters - 1)) * triu(nom / torch.sqrt(dnom_squared))
    return d


EPSILON = 1E-9
from torch.nn.functional import relu

def kernel_from_distance_matrix(dist, rel_sigma, min_sigma=EPSILON):
    """
    Compute a Gaussian kernel matrix from a distance matrix.

    :param dist: Disatance matrix
    :type dist: th.Tensor
    :param rel_sigma: Multiplication factor for the sigma hyperparameter
    :type rel_sigma: float
    :param min_sigma: Minimum value for sigma. For numerical stability.
    :type min_sigma: float
    :return: Kernel matrix
    :rtype: th.Tensor
    """
    # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
    dist = relu(dist)
    sigma2 = rel_sigma * torch.median(dist)
    # Disable gradient for sigma
    sigma2 = sigma2.detach()
    sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
    k = torch.exp(- dist / (2 * sigma2))
    return k


def vector_kernel(x, rel_sigma=0.15):
    """
    Compute a kernel matrix from the rows of a matrix.

    :param x: Input matrix
    :type x: th.Tensor
    :param rel_sigma: Multiplication factor for the sigma hyperparameter
    :type rel_sigma: float
    :return: Kernel matrix
    :rtype: th.Tensor
    """
    return kernel_from_distance_matrix(cdist(x, x), rel_sigma)


def cdist(X, Y):
    """
    Pairwise distance between rows of X and rows of Y.

    :param X: First input matrix
    :type X: th.Tensor
    :param Y: Second input matrix
    :type Y: th.Tensor
    :return: Matrix containing pairwise distances between rows of X and rows of Y
    :rtype: th.Tensor
    """
    xyT = X @ torch.t(Y)
    x2 = torch.sum(X**2, dim=1, keepdim=True)
    y2 = torch.sum(Y**2, dim=1, keepdim=True)
    d = x2 - 2 * xyT + torch.t(y2)
    return d


