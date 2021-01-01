import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from math import sqrt

import os
import json

from pprint import pprint
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--problems', type=str, help='file path to problems.json')
parser.add_argument('-s', '--submissions', type=str, help='file path to user_submissions.json')
parser.add_argument('-d', '--dir', type=str, help='dir to models')
parser.add_argument('-e', '--epochs', type=int, default=10, help='number of epochs to train for each BKT')
parser.add_argument('-f', '--fits', type=int, default=10, help='number of BKTs to train for each skill')
parser.add_argument('--forget', action='store_true', default=False, help='enable BKT to forget')
parser.add_argument('--restore', action='store_true', default=False, help='restore models from the dir')
parser.add_argument('--alpha', type=float, default=.05, help='adam-alpha')
parser.add_argument('--betas', type=float, nargs=2, default=[.9, .999], help='adam-betas')
parser.add_argument('-k', type=int, default=1, help='k-fold cross validation')
parser.add_argument('--seed', type=int, default=1, help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


class BKT(nn.Module):

    def __init__(self, forgetting=True):
        super(BKT, self).__init__()
        self.forgetting = forgetting
        self.L0 = nn.Parameter(torch.tensor(np.random.randn()), requires_grad=True)
        self.T = nn.Parameter(torch.tensor(np.random.randn()), requires_grad=True)
        if forgetting:
            self.F = nn.Parameter(torch.tensor(np.random.randn()), requires_grad=True)
        self.G = nn.Parameter(torch.tensor(np.random.randn()), requires_grad=True)
        self.S = nn.Parameter(torch.tensor(np.random.randn()), requires_grad=True)

    def forward(self, x):
        """
        :param x: (num_action, batch_size)
        :return: (num_pred, batch_size)
        """
        trans = torch.sigmoid(self.T)
        if self.forgetting:
            forget = torch.sigmoid(self.F)
        else:
            forget = torch.tensor(0.)
        guess = torch.sigmoid(self.G)
        slip = torch.sigmoid(self.S)
        one = torch.ones(x.size(1))
        learn = one * torch.sigmoid(self.L0)
        y = torch.zeros(x.size())
        for t in range(x.size(0)):
            # P(correct(t)) = P(L(t)) * (1 - P(S)) + (1 - P(L(t))) * P(G)
            correct = learn * (one - slip) + (one - learn) * guess
            y[t] = correct
            # action = correct:
            #     P(L(t)|correct(t)) = (P(L(t)) * (1 - P(S))) / P(correct(t))
            # action = incorrect
            #     P(L(t)|incorrect(t)) = (P(L(t)) * P(S)) / P(incorrect(t))
            conditional_probability = x[t] * (learn * (one - slip) / correct) \
                                      + (one - x[t]) * (learn * slip / (one - correct))
            # P(L(t+1)) = P(L(t)|action(t)) + (1 - P(L(t)|action(t))) * P(T)
            learn = conditional_probability * (one - forget) + (one - conditional_probability) * trans
        return y


def fit(x, mask, num_epochs=args.epochs, lr=args.alpha, betas=args.betas, num_fit=args.fits,
        forgetting=args.forget, restore_model=args.restore,
        test_x=None, test_mask=None, title=None):
    """
    randomly initialize num_fit BKT models,
    use Adam to optimize the MSE loss function,
    the training set is used to estimate the parameters,
    the best estimation is the one with the highest
    ROC AUC score on the training set,
    the prediction for the test set
    and the best estimated parameters are returned.

    :param x: training set, sized (num_action, batch)
    :param mask: training set mask, sized (num_action, batch)
    :param num_epochs: for each BKT, the number of epochs used to train the model
    :param lr: learning rate for optimizer
    :param num_fit: number of random initialized BKTs to estimate parameters
    :param forgetting: whether to enable forgetting in BKT
    :param restore_model: whether to restore model if model exists
    :param test_x: test set, sized (num_action, batch)
    :param test_mask: test set mask, sized (num_action, batch)
    :param title: the name of the model (a.k.a. the chart title)
    :return: the prediction for the test set, along with the model parameters
    """

    counter = 0

    best_bkt = None
    best_score = 0
    best_loss = None

    if title:
        model_path = os.path.join(args.dir, 'bkt - ' + title + (' - f' if forgetting else '') + '.pth')
    else:
        model_path = None

    if model_path and restore_model and os.path.exists(model_path):
        best_bkt = BKT(forgetting=forgetting)
        best_bkt.load_state_dict(torch.load(model_path))
    else:
        while counter != num_fit:
            counter += 1

            bkt = BKT(forgetting=forgetting)
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(bkt.parameters(), lr=lr, betas=betas)
            epoch = 0

            loss_list = []

            while epoch != num_epochs:
                epoch += 1
                bkt.train()
                with torch.enable_grad():
                    optimizer.zero_grad()
                    y = bkt(x)
                    loss = loss_fn(y * mask, x)
                    loss.backward()
                    nn.utils.clip_grad_norm_(bkt.parameters(), max_norm=2.)
                    optimizer.step()

                    loss_list.append(loss.item())

                print(counter, epoch, loss_list[-1])

            bkt.eval()
            with torch.no_grad():
                y = bkt(x) * mask

            y_true = x.masked_select(mask != 0).numpy()
            y_pred = y.masked_select(mask != 0).numpy()

            try:
                score = roc_auc_score(y_true, y_pred)
                if score > best_score:
                    best_bkt = bkt
                    best_score = score
                    best_loss = loss_list
            except ValueError as e:
                print('during fitting model %d:' % counter)
                print(e)
                print('refitting...')
                counter -= 1

        if model_path:
            torch.save(best_bkt.state_dict(), model_path)

    best_bkt.eval()
    with torch.no_grad():
        if test_x is not None and test_mask is not None:
            y = best_bkt(test_x)
        else:
            y = best_bkt(x)

    # if best_loss:
    #     plt.plot(best_loss)
    #     if title:
    #         plt.title(title)
    #     plt.show()

    return y * test_mask if test_x is not None and test_mask is not None else y * mask, {
        'prior': torch.sigmoid(best_bkt.L0).item(),
        'learn': torch.sigmoid(best_bkt.T).item(),
        'forget': torch.sigmoid(best_bkt.F).item() if forgetting else 0.,
        'guess': torch.sigmoid(best_bkt.G).item(),
        'slip': torch.sigmoid(best_bkt.S).item()
    }


with open(args.problems, 'r') as file:
    problems = json.load(file)
problem_id_2_tag_ids = {problem['id']: problem['tags'] for problem in problems}
tags = set()
for problem in problems:
    tags |= set(problem['tags'])
tags = list(sorted(list(tags)))
with open(args.submissions, 'r') as file:
    user_submissions = json.load(file)


def prepare_data(tag, training, group):
    ret_data = []
    ret_max_length = 0
    for user_data in user_submissions:
        user_group = user_data['group']
        if training and user_group == group \
                or not training and user_group != group:
            continue
        submissions = user_data['submissions']
        record = []
        for sub in submissions:
            if tag in problem_id_2_tag_ids[sub['problem']]:
                record.append(sub['verdict'])
        if len(record):
            ret_data.append(record)
            ret_max_length = max(ret_max_length, len(record))
    return ret_data, ret_max_length


def convert(data, data_max_length):
    batch_size = len(data)
    ret_x = np.zeros((data_max_length, batch_size))
    ret_mask = np.zeros((data_max_length, batch_size))
    for idx in range(batch_size):
        for i in range(len(data[idx])):
            ret_x[i][idx] = data[idx][i]
            ret_mask[i][idx] = 1
    return torch.tensor(ret_x), torch.tensor(ret_mask)


def run(group):
    y_true = np.zeros(0)
    y_pred = np.zeros(0)

    for tag in tags:
        train, train_max_length = prepare_data(tag, training=True, group=group)
        test, test_max_length = prepare_data(tag, training=False, group=group)

        if train_max_length and test_max_length:
            print("data set for '%d' has been prepared" % tag)
            train_x, train_mask = convert(train, train_max_length)
            test_x, test_mask = convert(test, test_max_length)
            print(train_x.shape, test_x.shape)

            print('fitting')
            test_y, params = fit(x=train_x, mask=train_mask,
                                 test_x=test_x, test_mask=test_mask,
                                 title=str(tag) + ' - ' + str(group))
            # pprint(params)

            y_true_part = test_x.masked_select(test_mask != 0).numpy()
            y_pred_part = test_y.masked_select(test_mask != 0).numpy()

            y_true = np.concatenate([y_true, y_true_part])
            y_pred = np.concatenate([y_pred, y_pred_part])

            print("ROC AUC on '%d': %.10f" % (tag, roc_auc_score(y_true_part, y_pred_part)))

    auc = roc_auc_score(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print('ROC AUC: {}'.format(auc))
    print('RMSE: {}'.format(rmse))
    print('MAE: {}'.format(mae))
    return auc, rmse, mae


def main():
    k = args.k
    auc = np.zeros(k)
    rmse = np.zeros(k)
    mae = np.zeros(k)
    for i in range(k):
        print('group %d:' % i)
        auc[i], rmse[i], mae[i] = run(i)
        print('-' * 30)
    print('ROC AUC: {} (+/- {})'.format(auc.mean(), auc.std()))
    print('RMSE: {} (+/- {})'.format(rmse.mean(), rmse.std()))
    print('MAE: {} (+/- {})'.format(mae.mean(), mae.std()))


if __name__ == '__main__':
    main()
