# Dynamic Key-Value Memory Networks (DKVMN)
#
# Paper: Jiani Zhang, Xingjian Shi, Irwin King, Dit-Yan Yeung.
# Dynamic Key-Value Memory Networks for Knowledge Tracing.
# arXiv:1611.08108v2 [cs.AI] 17 Feb 2017
# 
# For further reference:
# https://github.com/jennyzhang0215/DKVMN
# 
# Paper: Chun-Kit Yeung.
# Deep-IRT: Make Deep Learning Based Knowledge Tracing Explainable Using Item Response Theory
# arXiv:1904.11738v1 [cs.LG] 26 Apr 2019


import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

import math
from math import sqrt
from time import time

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
        mask = torch.zeros(self.max_q)
        for i in range(len(self.students_data[idx])):
            sub = self.students_data[idx][i]
            problems[i] = sub['problem']
            interactions[i] = sub['verdict']
            mask[i] = 1
        return problems, interactions, mask


class Memory(nn.Module):

    def __init__(self, d_k, d_v, h, n, dropout=.1):
        """
        :param d_k: key dim
        :param d_v: value dim
        :param h: hidden layer
        :param n: number of latent concepts
        :param dropout: dropout probability
        """
        super(Memory, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n = n
        self.w_key = nn.Linear(d_k, n, bias=False)
        self.w_erase = nn.Linear(d_v, d_v)
        self.w_add = nn.Linear(d_v, d_v)
        self.w = nn.Linear(d_v + d_k, h)
        self.dropout = nn.Dropout(dropout)
        self.init_memory = nn.Parameter(torch.randn(n, d_v))
    
    def attention(self, problems):
        """
        :param problems: (batch_size, d_k)
        :return: (batch_size, n)
        """
        return torch.softmax(self.w_key(problems), dim=-1)
    
    def read(self, score, memory):
        """
        :param score: (batch_size, n)
        :param memory: (batch_size, n, d_v)
        :return: (batch_size, d_v)
        """
        # score = score.view(-1, 1, self.n)  # (batch_size, 1, n)
        score = score.view(-1, 1)
        memory = memory.contiguous().view(-1, self.d_v)
        result = (score * memory).view(-1, self.n, self.d_v)
        # return torch.matmul(score, memory).view(-1, self.d_v)  # (batch_size, 1, d_v) => (batch_size, d_v)
        return torch.sum(result, dim=1)

    def write(self, score, memory, interactions):
        """
        :param score: (batch_size, n)
        :param memory: (batch_size, n, d_v)
        :param interactions: (batch_size, d_v)
        :return: memory
        """
        e = torch.sigmoid(self.w_erase(interactions))  # (batch_size, d_v)
        memory = memory * (1 - torch.matmul(score.view(-1, self.n, 1), e.view(-1, 1, self.d_v)))
        a = torch.tanh(self.w_add(interactions))  # (batch_size, d_v)
        memory = memory + torch.matmul(score.view(-1, self.n, 1), a.view(-1, 1, self.d_v))
        return memory

    def forward(self, problems, interactions):
        """
        :params problems: (batch_size, len, d_k)
        :params interactions: (batch_size, len, d_v)
        :return: (batch_size, len, h)
        """
        assert problems.device == interactions.device
        batch_size = problems.size(0)
        l = problems.size(1)
        problems = problems.transpose(0, 1)  # (len, batch_size, d_k)
        interactions = interactions.transpose(0, 1)  # (len, batch_size, d_v)
        x = torch.zeros(l, batch_size, self.d_v + self.d_k, device=problems.device)
        memory = self.init_memory.expand([batch_size, self.n, self.d_v])
        for i in range(l):
            score = self.attention(problems[i])
            x[i] = torch.cat([self.read(score, memory), problems[i]], dim=-1)
            memory = self.write(score, memory, interactions[i])
        x = x.transpose(0, 1)  # (batch_size, len, d_v + d_k)
        return torch.tanh(self.w(self.dropout(x)))


class DKVMN(nn.Module):

    def __init__(self, num_problems,
                 d_k, d_v, h, n, dropout=.1, IRT=False):
        super(DKVMN, self).__init__()
        self.num_problems = num_problems
        self.memory = Memory(d_k, d_v, h, n, dropout)
        self.exercise_embedding = nn.Embedding(num_problems, d_k)
        self.interaction_embedding = nn.Embedding(num_problems * 2, d_v)
        self.dropout = nn.Dropout(dropout)
        self.IRT = IRT
        self.w_f = nn.Linear(h, 1)
        if IRT:
            self.w_q = nn.Linear(d_k, 1)

    def forward(self, problems, interactions):
        assert problems.device == interactions.device
        interactions = problems + interactions * self.num_problems
        problems = self.exercise_embedding(problems.long())
        interactions = self.interaction_embedding(interactions.long())
        x = self.memory(problems, interactions)  # (batch_size, len, h)
        x = self.w_f(self.dropout(x))  # (batch_size, len, 1)
        if self.IRT:
            b = torch.tanh(self.w_q(self.dropout(problems)))  # (batch_size, len, 1)
            assert x.size() == b.size()
            x = 3.0 * torch.tanh(x) - b
        return torch.sigmoid(x)


def output(auc, rmse, mae):
    print("ROC AUC: {}".format(auc))
    print("RMSE: {}".format(rmse))
    print("MAE: {}".format(mae))


def train(problems, submissions, model_dir, num, group,
          lr=.001, betas=(.9, .999), max_grad_norm=2.,
          n=200, d_k=256, d_v=512, h=2048, dropout=.1, patience=10, shuffle=False,
          num_epochs=30, batch_size=32, max_q=100, irt=False, cuda=True):

    model_name = os.path.join(model_dir, 'dkvmn - %d %d' % (num, group))

    dkvmn_model_path = model_name + '.pth'

    training_set = SkillDataset(problems_file=problems,
                                submissions_file=submissions,
                                training=True, group=group, max_q=max_q)

    training_set_loader = DataLoader(training_set,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=2)

    test_set = SkillDataset(problems_file=problems,
                            submissions_file=submissions,
                            training=False, group=group, max_q=max_q)

    test_set_loader = DataLoader(test_set,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=2)

    print('max skills: %d' % training_set.max_skill)

    model = DKVMN(training_set.max_skill, d_k, d_v, h, n, dropout=dropout, IRT=irt)

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
        print("Entering #%d, group %d, epoch %d:" % (num, group, epoch))
        model.train()
        with torch.enable_grad(), tqdm(total=len(training_set), ascii=True) as progress_bar:
            for problems, interactions, masks in training_set_loader:
                masks = (masks == 1)

                if cuda:
                    problems = problems.cuda()
                    interactions = interactions.cuda()
                    masks = masks.cuda()

                optimizer.zero_grad()

                batch_out = model(problems, interactions)[:, :, 0]

                assert batch_out.size() == interactions.size()
                assert batch_out.size() == masks.size()

                # loss = loss_fn(batch_out * masks, interactions)
                loss = loss_fn(batch_out.masked_select(masks), interactions.masked_select(masks))

                epoch_loss += loss.item()

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
            torch.save(model.state_dict(), dkvmn_model_path)
            best_auc = auc
            best_auc_epoch = epoch

        if epoch - best_auc_epoch >= patience:
            print('Early Stopping: No AUC improvement in the last %d epochs.' % patience)
            break

    plt.figure()
    plt.plot(loss_list)
    plt.savefig(model_name + '.svg')

    model.load_state_dict(torch.load(dkvmn_model_path))
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
    parser.add_argument('-b', '--batch', type=int, default=32, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('-d_k', '--key-dim', type=int, default=128, help='key dimension')
    parser.add_argument('-d_v', '--value-dim', type=int, default=256, help='value dimension')
    parser.add_argument('-H', '--hidden', type=int, default=1024, help='hidden layer size')
    parser.add_argument('-n', type=int, default=20, help='number of latent concepts')
    parser.add_argument('-l', type=int, default=200, help='max length')
    parser.add_argument('--deep-irt', action='store_true', default=False, help='use Deep-IRT model')
    parser.add_argument('--shuffle', action='store_true', default=False, help='ramdom shuffle training set data')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='use cpu only')
    parser.add_argument('--dropout', type=float, default=.1, help='dropout probability')
    parser.add_argument('--alpha', type=float, default=.001, help='adam-alpha')
    parser.add_argument('--betas', type=float, nargs=2, default=[.9, .999], help='adam-betas')
    parser.add_argument('--max-grad-norm', type=float, default=2., help='max grad norm allowed when clipping')
    parser.add_argument('--patience', type=int, default=10, help='number of epochs to wait if AUC does not improve')
    parser.add_argument('-r', '--repeat', type=int, default=1, help='times of repetition')
    parser.add_argument('-k', type=int, default=1, help='k-fold cross validation')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args(argv)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

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
                                                                    lr=args.alpha, betas=args.betas, max_grad_norm=args.max_grad_norm,
                                                                    n=args.n, d_k=args.key_dim, d_v=args.value_dim, h=args.hidden,
                                                                    dropout=args.dropout, patience=args.patience, shuffle=args.shuffle,
                                                                    batch_size=args.batch, num_epochs=args.epochs, max_q=args.l,
                                                                    irt=args.deep_irt, cuda=args.cuda)
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
