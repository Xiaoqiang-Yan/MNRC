import sys
import torch
import torch.nn as nn
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
    # loss = p_i_j * (torch.log(p_i_j) - torch.log(p_j) - torch.log(p_i))
    # loss = 1/(loss.sum() + EPS)
    return loss


def compute_joint(x_img, x_txt):
    bn, k = x_img.size()
    assert (x_txt.size(0) == bn and x_txt.size(1) == k)
    p_i_j = x_img.unsqueeze(2) * x_txt.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.
    p_i_j = p_i_j / p_i_j.sum()
    return p_i_j


def l2_loss(out1, out2, size, temperature):
    out = torch.cat([out1, out2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out1 * out2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    l2_loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return l2_loss


def mask_type_transfer(mask):
    mask = mask.type(torch.bool)
    # mask = mask.type(torch.uint8)
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
