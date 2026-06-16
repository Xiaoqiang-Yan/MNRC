import argparse
import torch
import numpy as np
import utils1
from mutual_information import mutual_information, l2_loss, NTXentLoss
from dataset import Dateset_mat, data_loder
from tqdm import trange, tqdm
from model import Encoder_f, MIEstimator, UD_constraint
import random
import warnings
import itertools

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", default=r'dataset/view4/flickr', type=str)  # coco 250, esp 190, flickr 35, iapr 420, nus 50
parser.add_argument("--lr", type=float, default=0.00005)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--fea_dim", type=int, default=128)
parser.add_argument("--temperature", type=int, default=0.05)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--view_num", type=int, default=4)
config = parser.parse_args()
config.max_ACC = 0

Dataset = Dateset_mat(config.dataset_root, config.view_num)
dataset = Dataset.getdata()
label1 = np.array(dataset[dataset.__len__()-1])-1
all_label = np.squeeze(label1)
cluster_num = max(all_label) + 1
print(config.dataset_root)
# print("clustering number: ", cluster_num)
# criterion = torch.nn.CrossEntropyLoss().to(device)
loss_NTXent = NTXentLoss(config.batch_size)

def run():
    max_ACC = 0
    all_label = np.squeeze(dataset[dataset.__len__()-1])
    if config.view_num == 3:
        all_v1 = torch.tensor(dataset[0], dtype=torch.float32).to(device)
        all_v2 = torch.tensor(dataset[1], dtype=torch.float32).to(device)
        all_v3 = torch.tensor(dataset[2], dtype=torch.float32).to(device)
        all_data = [all_v1, all_v2, all_v3, all_label]
        in_channel = [all_v1.size(1), all_v2.size(1), all_v3.size(1)]
    elif config.view_num == 2:
        all_v1 = torch.tensor(dataset[0], dtype=torch.float32).to(device)
        all_v2 = torch.tensor(dataset[1], dtype=torch.float32).to(device)
        all_data = [all_v1, all_v2, all_label]
        in_channel = [all_v1.size(1), all_v2.size(1)]
    elif config.view_num == 4:
        all_v1 = torch.tensor(dataset[0], dtype=torch.float32).to(device)
        all_v2 = torch.tensor(dataset[1], dtype=torch.float32).to(device)
        all_v3 = torch.tensor(dataset[2], dtype=torch.float32).to(device)
        all_v4 = torch.tensor(dataset[3], dtype=torch.float32).to(device)
        all_data = [all_v1, all_v2, all_v3, all_v4, all_label]
        in_channel = [all_v1.size(1), all_v2.size(1), all_v3.size(1), all_v4.size(1)]
    print("clustering number: ", cluster_num)
    data = data_loder(config.batch_size, config.view_num)
    data.get_data(dataset)
    model = Encoder_f(config.view_num, in_channel, config.fea_dim, cluster_num).to(device)
    mi_estimator = MIEstimator(config.fea_dim, config.fea_dim).to(device)
    parame = itertools.chain(model.parameters(), mi_estimator.parameters())
    optimiser = torch.optim.Adam(parame, lr=config.lr)

    for epoch in range(config.num_epochs):
        model.train()
        model.zero_grad()
        for data_ in data:
            data_ = [data_[i].to(device) for i in range(data_.__len__()-1)]

            P_F, P, clustering = model(data_)    # P_F融合后的分布 P每个编码器的分布 clustering每个编码器的聚类结果
            loss2 = getL2Loss(clustering)        # NTXnet
            loss1 = getUDLoss(clustering)        # UDC
            lossKL, lossMI = getMILoss(P_F, P, mi_estimator)  # KL
            loss = loss1 + loss2 + lossKL + 0.001 * lossMI
            # lossKL = getMILoss(P_F, P, mi_estimator)
            # loss = loss1 + loss2 + lossKL
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        if epoch % 2 == 0:
            acc, nmi = get_S_ACC(model, all_data, epoch)
            acc1 = max(acc)
            if acc1 > max_ACC:
                max_ACC = acc1
            if config.view_num == 3:
                print("epoch:%d acc1 %.4f nmi1 %.4f acc2 %.4f nmi2 %.4f, acc3 %.4f nmi3 %.4f max_acc %.4f "
                      % (epoch, acc[0], nmi[0], acc[1], nmi[1], acc[2], nmi[2], max_ACC))
                # print("S: loss1 %.4f loss2 %.4f lossKL %.4f lossMI %.4f" % (loss1, loss2, lossKL, lossMI))
            if config.view_num == 2:
                print("epoch:%d: acc1 %.4f nmi1 %.4f acc2 %.4f nmi2 %.4f,max_acc %.4f "
                      % (epoch, acc[0], nmi[0], acc[1], nmi[1], max_ACC))
                # print("S: loss1 %.4f loss2 %.4f lossKL %.4f lossMI %.4f" % (loss1, loss2, lossKL, lossMI))
            if config.view_num == 4:
                print("epoch:%d: acc1 %.4f nmi1 %.4f acc2 %.4f nmi2 %.4f \n acc3 %.4f nmi3 %.4f acc4 %.4f nmi4 %.4f, \n max_acc %.4f "
                    % (epoch, acc[0], nmi[0], acc[1], nmi[1], acc[2], nmi[2], acc[3], nmi[3], max_ACC))
                # print("S: loss1 %.4f loss2 %.4f lossKL %.4f lossMI %.4f" % (loss1, loss2, lossKL, lossMI))
    del data
    return max_ACC


def getL2Loss(clusterings):
    matrix = np.triu(np.ones((config.view_num, config.view_num)), 1)
    Index = list(zip(*np.nonzero(matrix)))
    loss = 0
    for index in Index:
        a, b = clusterings[index[0]], clusterings[index[1]]
        loss += loss_NTXent(a, b)
        # loss += l2_loss(a, b, a.size(0), config.temperature)

    return loss


def getUDLoss(clusterings):
    loss = 0
    criterion = torch.nn.CrossEntropyLoss().to(device)
    for clustering in clusterings:
        loss += criterion(clustering, UD_constraint(clustering).to(device))
    return loss

def getMILoss(P_F, P, mi_estimator):
    x_P_F = P_F.rsample()
    loss1 = 0
    loss2 = 0
    for p in P:
        x_p = p.rsample()
        miG, _ = mi_estimator(x_P_F, x_p)
        kl_1_2 = P_F.log_prob(x_P_F) - p.log_prob(x_P_F)
        kl_2_1 = p.log_prob(x_p) - P_F.log_prob(x_p)
        skl = (kl_1_2 + kl_2_1).mean() / 2.
        loss2 += mutual_information(x_p, x_P_F)
        loss1 += -miG + 0.005*skl
    return loss1, loss2
    # return loss2


def get_S_ACC(model, all_data, epoch):
    model.eval()
    label = all_data[all_data.__len__()-1]
    # all_data.pop()
    _, _, x_out = model(all_data[:all_data.__len__()-1])
    acc, nmi = [], []
    for clustering in x_out:
        pre_label = np.array(clustering.cpu().detach().numpy())
        pre_label = np.argmax(pre_label, axis=1)

        acc.append(utils1.metrics.acc(pre_label, label))
        nmi.append(utils1.metrics.nmi(pre_label, label))

    return acc, nmi


def fix_seed(seed):
    print('seed:', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    fix_seed(645)
    run()


