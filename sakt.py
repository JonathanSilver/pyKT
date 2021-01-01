# Self-Attentive Knowledge Tracing (SAKT)
#
# Paper: Shalini Pandey, George Karypis.
# A Self-Attentive model for Knowledge Tracing.
# arXiv:1907.06837v1 [cs.LG] 16 Jul 2019
#
# Code is adapted from:
# http://nlp.seas.harvard.edu/2018/04/03/attention.html
# 
# For further reference:
# Paper: Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.
# Attention Is All You Need.
# arXiv:1706.03762v5 [cs.CL] 6 Dec 2017


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

import math
from math import sqrt
from time import time

import copy
import os
import sys
import argparse

import json


class SkillDataset(Dataset):
    
    def __init__(self, problems_file, submissions_file, training, group, max_q):
        super(SkillDataset, self).__init__()
        with open(problems_file, 'r') as file:
            problem_data = json.load(file)
        with open(submissions_file, 'r') as file:
            user_submissions = json.load(file)
        self.max_skill = len(problem_data)
        self.max_q = max_q
        self.students_data = []
        for user_data in user_submissions:
            user_group = user_data['group']
            if training and user_group == group \
                    or not training and user_group != group:
                continue
            submissions = user_data['submissions']
            num_submissions = len(submissions)
            res = [submissions[k:min(k + max_q, num_submissions)]
                   for k in range(0, num_submissions, max_q)]
            self.students_data += filter(lambda x: len(x) > 1, res)

    def __len__(self):
        return len(self.students_data)

    def __getitem__(self, idx):
        problems = torch.zeros(self.max_q)
        interactions = torch.zeros(self.max_q)
        masks = torch.zeros(self.max_q)
        for i in range(len(self.students_data[idx])):
            sub = self.students_data[idx][i]
            problems[i] = sub['problem']
            interactions[i] = sub['verdict']
            masks[i] = 1
        return problems, interactions, masks


## [Copy & Paste] begin.


def clones(module, n):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class LayerNorm(nn.Module):
    """Construct a layernorm module."""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return self.norm(x + sublayer(self.dropout(x)))


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


def subsequent_mask(size, device):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    _subsequent_mask = np.triu(np.ones(attn_shape), k=0).astype('uint8')
    return torch.tensor(_subsequent_mask, device=device) == 0


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(n_batches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


## [Copy & Paste] end.


class SAKT(nn.Module):

    def __init__(self, num_problems, max_len, N=6,
                 d_model=512, d_ff=2048, h=8, dropout=.1):
        super(SAKT, self).__init__()
        c = copy.deepcopy
        self.N = N
        self.num_problems = num_problems
        # for the first layer
        self.attn = MultiHeadedAttention(h, d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer = SublayerConnection(d_model, dropout)
        if N > 1:
            # for the N - 1 encoder layers
            self.encoder = Encoder(EncoderLayer(d_model, c(self.attn), c(self.ff), dropout), N - 1)
        # for the last generator
        self.generator = nn.Linear(d_model, 2)
        # for the embeddings
        self.interaction_embedding = Embeddings(d_model, num_problems * 2)
        self.position_embedding = Embeddings(d_model, max_len)
        self.exercise_embedding = Embeddings(d_model, num_problems)

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, problems, interactions):
        assert problems.device == interactions.device
        interactions = problems + interactions * self.num_problems
        n = problems.size(1)
        positions = torch.arange(n, device=problems.device).long()
        problems = self.exercise_embedding(problems.long())
        interactions = self.interaction_embedding(interactions.long()) + self.position_embedding(positions)
        mask = subsequent_mask(n, problems.device)
        out = self.attn(problems, interactions, interactions, mask)
        out = self.layer(out, self.ff)
        if self.N > 1:
            out = self.encoder(out, mask)
        return torch.softmax(self.generator(out), dim=-1)


def output(auc, rmse, mae):
    print("ROC AUC: {}".format(auc))
    print("RMSE: {}".format(rmse))
    print("MAE: {}".format(mae))


def train(problems, submissions, model_dir, num, group,
          lr=.001, betas=(.9, .999), n=6, dim=512, hidden=2048, heads=8, dropout=.1,
          patience=10, num_epochs=30, batch_size=32, max_q=100, cuda=True):

    model_name = os.path.join(model_dir, 'sakt - %d %d' % (num, group))

    sakt_model_path = model_name + '.pth'

    training_set = SkillDataset(problems_file=problems,
                                submissions_file=submissions,
                                training=True, group=group, max_q=max_q)

    training_set_loader = DataLoader(training_set,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=2)

    test_set = SkillDataset(problems_file=problems,
                            submissions_file=submissions,
                            training=False, group=group, max_q=max_q)

    test_set_loader = DataLoader(test_set,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=2)

    print('max skills: %d' % training_set.max_skill)

    model = SAKT(training_set.max_skill, max_q, N=n, d_model=dim, d_ff=hidden, h=heads, dropout=dropout)

    if cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)

    loss_fn = nn.BCELoss()
    loss_list = []

    best_auc = 0
    best_auc_epoch = 0

    epoch = 0
    while epoch != num_epochs:
        epoch += 1
        epoch_loss = 0
        print("Entering group %d, epoch %d:" % (group, epoch))
        model.train()
        with torch.enable_grad(), tqdm(total=len(training_set), ascii=True) as progress_bar:
            for problems, interactions, masks in training_set_loader:
                if cuda:
                    problems = problems.cuda()
                    interactions = interactions.cuda()
                    masks = masks.cuda()

                optimizer.zero_grad()

                batch_out = model(problems, interactions)[:, :, 0]
                assert batch_out.size() == interactions.size()
                loss = loss_fn(batch_out * masks, interactions)

                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

                progress_bar.update(problems.size(0))
                progress_bar.set_postfix(epoch=epoch, loss=loss.item())

        loss_list.append(epoch_loss)
        print("Epoch loss: {}".format(epoch_loss))

        # print("Evaluating the trained model (training set)")
        # auc, rmse, mae = evaluate(model, training_set, training_set_loader, cuda)
        # output(auc, rmse, mae)
        print("Evaluating the trained model")
        auc, rmse, mae = evaluate(model, test_set, test_set_loader, cuda)
        output(auc, rmse, mae)
        print("Evaluation complete")

        if auc > best_auc:
            torch.save(model.state_dict(), sakt_model_path)
            best_auc = auc
            best_auc_epoch = epoch

        if epoch - best_auc_epoch >= patience:
            print('Early Stopping: No AUC improvement in the last %d epochs.' % patience)
            break

    plt.figure()
    plt.plot(loss_list)
    plt.savefig(model_name + '.svg')

    model.load_state_dict(torch.load(sakt_model_path))
    auc, rmse, mae = evaluate(model, test_set, test_set_loader, cuda)
    print('*' * 30)
    print('Best Model: %d' % best_auc_epoch)
    output(auc, rmse, mae)
    return auc, rmse, mae


def evaluate(model, test_set, test_set_loader, cuda):
    y_true = torch.zeros(0)
    y_pred = torch.zeros(0)

    if cuda:
        y_true = y_true.cuda()
        y_pred = y_pred.cuda()

    model.eval()
    with torch.no_grad(), tqdm(total=len(test_set), ascii=True) as progress_bar:
        for problems, interactions, masks in test_set_loader:
            masks = (masks == 1)
            if cuda:
                problems = problems.cuda()
                interactions = interactions.cuda()
                masks = masks.cuda()

            batch_out = model(problems, interactions)[:, :, 0]

            y_true = torch.cat([y_true, interactions.masked_select(masks)])
            y_pred = torch.cat([y_pred, batch_out.masked_select(masks)])

            progress_bar.update(problems.size(0))

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    return roc_auc_score(y_true, y_pred), \
        sqrt(mean_squared_error(y_true, y_pred)), \
        mean_absolute_error(y_true, y_pred)


def main(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--problems', type=str, help='file path to problems.json')
    parser.add_argument('-s', '--submissions', type=str, help='file path to user_submissions.json')
    parser.add_argument('-D', '--dir', type=str, help='dir to models')
    parser.add_argument('-b', '--batch', type=int, default=128, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('-d', '--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('-H', '--hidden', type=int, default=256, help='hidden layer size')
    parser.add_argument('--heads', type=int, default=8, help='number of heads')
    parser.add_argument('-n', type=int, default=3, help='number of layers')
    parser.add_argument('-l', type=int, default=100, help='max length')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='use cpu only')
    parser.add_argument('--dropout', type=float, default=.1, help='dropout probability')
    parser.add_argument('--alpha', type=float, default=.001, help='adam-alpha')
    parser.add_argument('--betas', type=float, nargs=2, default=[.9, .999], help='adam-betas')
    parser.add_argument('--patience', type=int, default=10, help='number of epochs to wait if AUC does not improve')
    parser.add_argument('-r', '--repeat', type=int, default=1, help='times of repetition')
    parser.add_argument('-k', type=int, default=1, help='k-fold cross validation')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args(argv)

    args.cuda = torch.cuda.is_available() and not args.no_cuda

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    k = args.k
    r = args.repeat
    start_time = time()
    
    auc = np.zeros(r * k)
    rmse = np.zeros(r * k)
    mae = np.zeros(r * k)
    for j in range(r):
        print('#%d:' % j)
        for i in range(k):
            print('group %d: ' % i)
            auc[j * k + i], rmse[j * k + i], mae[j * k + i] = train(args.problems, args.submissions, args.dir, j, i,
                                                                    n=args.n, dim=args.dim, hidden=args.hidden, heads=args.heads,
                                                                    lr=args.alpha, betas=args.betas, dropout=args.dropout, patience=args.patience,
                                                                    batch_size=args.batch, num_epochs=args.epochs, max_q=args.l, cuda=args.cuda)
            print('-' * 30)
    print()
    print('=' * 30)
    print('ROC AUC: {} (+/- {})'.format(auc.mean(), auc.std()))
    print('RMSE: {} (+/- {})'.format(rmse.mean(), rmse.std()))
    print('MAE: {} (+/- {})'.format(mae.mean(), mae.std()))
    print('=' * 30)
    print()
    print()

    print('Elapsed time: ' + str(time() - start_time))


if __name__ == '__main__':
    main(sys.argv[1:])
