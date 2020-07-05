from __future__ import print_function
import torch
import torch.nn.functional as F
import numpy as np


def n_peak_find(x, args, win_size):
    # x : (300,20)
    split_list = [win_size] * (x.shape[0] // win_size - 1) + [
        x.shape[0] - max(0, win_size * (x.shape[0] // win_size - 1))
    ]
    splits = torch.split(x, split_list, dim=0)  # [(60,20),(60,20),(60,20),(60,20),(60,20)]
    peak = torch.Tensor()
    for x_split in splits:
        sp_topk = torch.topk(x_split, int(min(x.shape[0], args.topk2)), dim=0)[
            0
        ]  # (10,20)
        mean_sp = torch.mean(sp_topk, 0, keepdim=True)  # (1,20)
        peak = torch.cat((peak, mean_sp))
    return peak   # (5,20)


def mill_loss(element_logits, seq_len, labels, device, args):
    k = np.ceil(seq_len / args.topk).astype("int32")
    eps = 1e-8
    loss = 0
    element_logits = F.hardtanh(element_logits, -args.clip, args.clip)  # args.clip = 4
    for i in range(element_logits.shape[0]):
        peaks = n_peak_find(
            element_logits[i][: seq_len[i]], args, win_size=int(args.topk)
        )

        block_peak = torch.sigmoid(peaks)   # (5,20)

        prob = 1 - torch.prod(1 - block_peak, 0)   # (20)

        lab = labels[i]
        loss1 = -torch.sum(lab * torch.log(prob + eps)) / torch.sum(lab)
        loss2 = -torch.sum((1 - lab) * torch.log(1 - prob + eps)) / torch.sum(
            1 - lab
        )
        loss += 1 / 2 * (loss1 + loss2)

    milloss = loss / element_logits.shape[0]  # 1.9133
    return milloss

# 对attention后的特征归一化
def get_unit_vector(x, dim=0):
    return x / torch.norm(x, 2, dim=dim, keepdim=True)


def batch_per_dis(X1, X2, w):  # (2048,4,1), (2048,4,1)  (1,2048)
    X1 = X1.permute(2, 1, 0).unsqueeze(2)  # (2048,4,1)-->(1,4,1,2048)
    X2 = X2.permute(2, 1, 0).unsqueeze(1)  # (2048,4,1)-->(1,1,4,2048)

    X_d = X1 - X2  # (1,4,4,2048)
    X_diff = X_d.reshape(X_d.shape[0], X_d.shape[1] * X_d.shape[2], -1)  # (1,16,2048)

    w = w.unsqueeze(-1)  # (1,2048,1)
    dis_mat = torch.bmm(X_diff, w).squeeze(-1)  # (1,16)
    dis_mat = dis_mat.reshape(dis_mat.shape[0], X_d.shape[1], X_d.shape[2])  # (1,4,4)
    dis_mat = torch.pow(dis_mat, 2)  # (1,4,4)
    return dis_mat


def metric_loss(
    x, element_logits, weight, labels, seq_len, device, args
):  # x:(20,300,2048)  element_logits:(20,300,20) weight : (20,2048) labels:(20,20)

    sim_loss = 0.0
    labels = labels
    n_tmp = 0.0

    element_logits = F.hardtanh(element_logits, -args.clip, args.clip)
    for i in range(0, args.num_similar * args.similar_size, args.similar_size):

        lab = labels[i, :]
        for k in range(i + 1, i + args.similar_size):
            lab = lab * labels[k, :]

        common_ind = lab.nonzero().squeeze(-1)  # [10]

        Xh = torch.Tensor()   # (2048,4,1)
        Xl = torch.Tensor()   # (2048,4,1)

        for k in range(i, i + args.similar_size):
            elem = element_logits[k][: seq_len[k], common_ind]  # (249,1)
            atn = F.softmax(elem, dim=0)  # (249,1)
            n1 = torch.FloatTensor([np.maximum(seq_len[k] - 1, 1)]).to(device)  # [248,]
            atn_l = (1 - atn) / n1  # (249,1)

            xh = torch.mm(torch.transpose(x[k][: seq_len[k]], 1, 0), atn)   # (249,2048)-->(2048,249)x(249,1)-->(2048,1)
            xl = torch.mm(torch.transpose(x[k][: seq_len[k]], 1, 0), atn_l) # (249,2048)-->(2048,249)x(249,1)-->(2048,1)
            xh = xh.unsqueeze(1)  # (2048,1)-->(2048,1,1)
            xl = xl.unsqueeze(1)  # (2048,1)-->(2048,1,1)
            Xh = torch.cat([Xh, xh], dim=1)
            Xl = torch.cat([Xl, xl], dim=1)

        Xh = get_unit_vector(Xh, dim=0)  # (2048,4,1)
        Xl = get_unit_vector(Xl, dim=0)  # (2048,4,1)

        D1 = batch_per_dis(Xh, Xh, weight[common_ind, :])  # (1,4,4)

        D1 = torch.triu(D1, diagonal=1)
        D1 = D1.view(D1.shape[0], -1)  # (1,16)
        d1 = torch.sum(D1, -1) / (
            args.similar_size * (args.similar_size - 1) / 2
        )  # 0.0002

        D2 = batch_per_dis(Xh, Xl, weight[common_ind, :])  #(1,4,4)
        """
        1 - torch.eye(D2.shape[1]):
            0 1 1 1
            1 0 1 1
            1 1 0 1
            1 1 1 0
        """

        D2 = D2 * (1 - torch.eye(D2.shape[1])).unsqueeze(0)  # (1,4,4)  
        D2 = D2.view(D2.shape[0], -1)

        d2 = torch.sum(D2, -1) / (args.similar_size * (args.similar_size - 1))  # 0.0012

        loss = torch.sum(
            torch.max(d1 - d2 + args.dis, torch.FloatTensor([0.0]).to(device))
        )
        sim_loss += loss
        n_tmp = n_tmp + torch.sum(lab)

    sim_loss = sim_loss / n_tmp
    return sim_loss



def train(
    itr, dataset, args, model, optimizer, logger, device, scheduler=None
):
    model.train()

    features, labels = dataset.load_data(
        n_similar=args.num_similar, similar_size=args.similar_size
    )
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, : np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    final_features, element_logits = model(features)

    loss_mill = mill_loss(element_logits, seq_len, labels, device, args)

    weight = model.classifier.weight
    loss_metric = metric_loss(
        final_features,
        element_logits,
        weight,
        labels,
        seq_len,
        device,
        args,
    )

    total_loss = args.Lambda * loss_mill + (1 - args.Lambda) * loss_metric

    logger.log_value("milloss", loss_mill, itr)
    logger.log_value("metricloss", loss_metric, itr)
    logger.log_value("total_loss", total_loss, itr)

    print("Iteration: %d, Loss: %.3f" % (itr, total_loss.data.cpu().numpy()))

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.data.cpu().numpy()
