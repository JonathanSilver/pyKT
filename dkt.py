# Deep Knowledge Tracing (Plus)
# 
# Paper: Chun-Kit Yeung, Dit-Yan Yeung
# Addressing Two Problems in Deep Knowledge Tracing via Prediction-Consistent Regularization
# arXiv:1806.02180v1 [cs.AI] 6 Jun 2018
# 
# 
# For further reference:
# 
# 1. Paper: Chris Piech, Jonathan Spencer, Jonathan Huang, et al.
# Deep Knowledge Tracing
# https://arxiv.org/abs/1506.05908
# 
# 2. Paper: Chun-Kit Yeung, Zizheng Lin, Kai Yang, et al.
# Incorporating Features Learned by an Enhanced Deep Knowledge Tracing Model for STEM/Non-STEM Job Prediction
# arXiv:1806.03256v1 [cs.CY] 6 Jun 2018


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from math import sqrt
from time import time

# import seaborn as sns

import json

import os
import sys
import argparse


class SkillLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout=.1):
        super(SkillLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths, hidden=None):
        mask = torch.zeros(x.size(0), x.size(1), x.size(2) // 2, device=x.device)
        for idx in range(mask.size(0)):
            mask[idx][:lengths[idx]] = 1
        orig_len = x.size(1)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        if hidden is not None:
            x, (hn, cn) = self.rnn(x, hidden)
        else:
            x, (hn, cn) = self.rnn(x)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        x = torch.sigmoid(self.linear(self.dropout(x)))
        return x * mask, (hn, cn)


class SkillDataset(Dataset):
    
    def __init__(self, problems_file, submissions_file, training, group, max_q=0, skills=True):
        super(SkillDataset, self).__init__()
        with open(problems_file, 'r') as file:
            problem_data = json.load(file)
        with open(submissions_file, 'r') as file:
            user_submissions = json.load(file)
        if skills:
            tags = set()
            for problem in problem_data:
                tags |= set(problem['tags'])
            self.max_skill = len(tags)
        else:
            self.max_skill = len(problem_data)
        self.skills = skills
        self.problem_id_2_tags = {problem['id']: problem['tags'] for problem in problem_data}
        self.max_q = max_q
        self.students_data = []
        for user_data in user_submissions:
            user_group = user_data['group']
            if training and user_group == group \
                    or not training and user_group != group:
                continue
            submissions = user_data['submissions']
            num_submissions = len(submissions)
            if max_q == 0:
                self.max_q = max(self.max_q, num_submissions)
                self.students_data.append(submissions)
            else:
                res = [submissions[k:min(k + max_q, num_submissions)]
                    for k in range(0, num_submissions, max_q)]
                self.students_data += filter(lambda x: len(x) > 1, res)

    def __len__(self):
        return len(self.students_data)

    def __getitem__(self, idx):
        submission = torch.zeros(self.max_q, 2 * self.max_skill)
        for i in range(len(self.students_data[idx])):
            sub = self.students_data[idx][i]
            if self.skills:
                for tag in self.problem_id_2_tags[sub['problem']]:
                    submission[i][tag] = 1
                    if sub['verdict'] == 1:
                        submission[i][self.max_skill + tag] = 1
            else:
                problem_id = sub['problem']
                submission[i][problem_id] = 1
                if sub['verdict'] == 1:
                    submission[i][self.max_skill + problem_id] = 1
        return submission, torch.tensor(len(self.students_data[idx]))


def output(auc, rmse, mae):
    print("ROC AUC: {}".format(auc))
    print("RMSE: {}".format(rmse))
    print("MAE: {}".format(mae))


def train(problems, submissions, model_dir, num, group,
          lambda_o, lambda_w1, lambda_w2, hidden_size, dropout=.1,
          lr=.001, betas=(.9, .999), max_grad_norm=2., patience=10,
          num_epochs=30, batch_size=32, max_q=1000, skills=True, dump=False,
          shuffle=False, compact=False):

    model_name = os.path.join(model_dir, ('dkt - %d %d %.1f %.1f %.1f' % (num, group, lambda_o, lambda_w1, lambda_w2)) + (' - skills' if skills else ''))

    dkt_model_path = model_name + '.pth'

    training_set = SkillDataset(problems_file=problems, submissions_file=submissions,
                                training=True, group=group, max_q=max_q, skills=skills)

    training_set_loader = DataLoader(training_set,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=2)

    test_set = SkillDataset(problems_file=problems, submissions_file=submissions,
                            training=False, group=group, max_q=max_q, skills=skills)

    test_set_loader = DataLoader(test_set,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=2)

    print('max skills: %d' % training_set.max_skill)

    model = SkillLSTM(training_set.max_skill * 2, hidden_size, training_set.max_skill, dropout)

    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)

    loss_bce = nn.BCELoss()
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
            for student, lengths in training_set_loader:
                student = student.cuda()

                optimizer.zero_grad()

                batch_out, _ = model(student, lengths)

                loss = torch.tensor(0, dtype=torch.float).cuda()

                if compact:
                    student_0 = student[:, :, :training_set.max_skill]
                    student_1 = student[:, :, training_set.max_skill:]

                    assert batch_out.size() == student_0.size()
                    assert batch_out.size() == student_1.size()

                    mask_next = (student_0[:, 1:] != 0)
                    loss += loss_bce(batch_out[:, :-1].masked_select(mask_next),
                                     student_1[:, 1:].masked_select(mask_next))
                    mask_curr = (student_0 != 0)
                    loss += lambda_o * loss_bce(batch_out.masked_select(mask_curr),
                                                student_1.masked_select(mask_curr))
                    
                    diff = batch_out[:, 1:] - batch_out[:, :-1]
                    loss += lambda_w1 * torch.mean(torch.abs(diff))
                    loss += lambda_w2 * torch.mean(diff ** 2)
                else:
                    for batch_idx in range(student.size(0)):
                        batch_out_part = batch_out[batch_idx][:lengths[batch_idx]]
                        student_part = student[batch_idx][:lengths[batch_idx]]
                        student_part_0 = student_part[:, :training_set.max_skill]
                        student_part_1 = student_part[:, training_set.max_skill:]

                        assert batch_out_part.size() == student_part_0.size()
                        assert batch_out_part.size() == student_part_1.size()

                        loss += loss_bce(batch_out_part[:-1] * student_part_0[1:], student_part_1[1:])
                        loss += lambda_o * loss_bce(batch_out_part * student_part_0, student_part_1)

                        diff = batch_out_part[1:] - batch_out_part[:-1]
                        loss += lambda_w1 * torch.mean(torch.abs(diff))
                        loss += lambda_w2 * torch.mean(diff ** 2)

                epoch_loss += loss.item()

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

                progress_bar.update(student.size(0))
                progress_bar.set_postfix(epoch=epoch, loss=loss.item())

        loss_list.append(epoch_loss)
        print("Epoch loss: {}".format(epoch_loss))

        print("Evaluating the trained model")
        auc, rmse, mae = evaluate(model, test_set, test_set_loader)
        output(auc, rmse, mae)
        print("Evaluation complete")

        if auc > best_auc:
            torch.save(model.state_dict(), dkt_model_path)
            best_auc = auc
            best_auc_epoch = epoch
        
        if epoch - best_auc_epoch >= patience:
            print('Early Stopping: No AUC improvement in the last %d epochs.' % patience)
            break

    plt.figure()
    plt.plot(loss_list)
    plt.savefig(model_name + '.svg')

    model.load_state_dict(torch.load(dkt_model_path))
    model.cuda()
    auc, rmse, mae = evaluate(model, test_set, test_set_loader)
    print('*' * 30)
    print('Best Model: %d' % best_auc_epoch)
    output(auc, rmse, mae)

    if skills and dump:
        print('+' * 30)
        print('Dumping user profiles')
        dataset = SkillDataset(problems_file=problems,
                               submissions_file=submissions,
                               training=True, group=-1, max_q=0, skills=True)
        user_profiles = []

        model.eval()
        with torch.no_grad(), tqdm(total=len(dataset), ascii=True) as progress_bar:
            for student, length in dataset:
                student = student.cuda()
                batch_out, _ = model(student.unsqueeze(0), length.unsqueeze(0))
                batch_out = batch_out[0]
                user_profiles.append(batch_out[:length].cpu().numpy())
                progress_bar.update(1)

        print('Total:', len(user_profiles))

        with open(model_name + ' - profiles.bin', 'wb') as file:
            import pickle
            pickle.dump(user_profiles, file)

    return auc, rmse, mae


def evaluate(model, test_set, test_set_loader):
    y_true = torch.zeros(0).cuda()
    y_pred = torch.zeros(0).cuda()

    model.eval()
    with torch.no_grad(), tqdm(total=len(test_set), ascii=True) as progress_bar:
        for student, lengths in test_set_loader:
            student = student.cuda()

            batch_out, _ = model(student, lengths)

            y_true_0 = student[:, 1:, :test_set.max_skill]
            y_true_1 = student[:, 1:, test_set.max_skill:]
            batch_out = batch_out[:, :-1]
            
            assert batch_out.size() == y_true_0.size()
            assert batch_out.size() == y_true_1.size()

            mask = (y_true_0 != 0)
            y_true = torch.cat([y_true, y_true_1.masked_select(mask)])
            y_pred = torch.cat([y_pred, batch_out.masked_select(mask)])

            progress_bar.update(student.size(0))

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    return roc_auc_score(y_true, y_pred), \
        sqrt(mean_squared_error(y_true, y_pred)), \
        mean_absolute_error(y_true, y_pred)


def main(argv):
    # sns.set()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--problems', type=str, help='file path to problems.json')
    parser.add_argument('-s', '--submissions', type=str, help='file path to user_submissions.json')
    parser.add_argument('-D', '--dir', type=str, help='dir to models')
    parser.add_argument('-b', '--batch', type=int, default=32, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('-H', '--hidden', type=int, default=64, help='DKT hidden layer size')
    parser.add_argument('-l', type=int, default=200, help='max length')
    parser.add_argument('-o', type=float, nargs='+', default=[0.], help='lambda_o')
    parser.add_argument('-w1', type=float, nargs='+', default=[0.], help='lambda_w1')
    parser.add_argument('-w2', type=float, nargs='+', default=[0.], help='lambda_w2')
    parser.add_argument('--dropout', type=float, default=.1, help='dropout probability')
    parser.add_argument('--skills', action='store_true', default=False, help='train skills DKT instead of standard DKT (use skill-level tags instead of exercise-level tags)')
    parser.add_argument('--dump', action='store_true', default=False, help='dump user profiles for skills DKT')
    parser.add_argument('--shuffle', action='store_true', default=False, help='random shuffle training set data')
    parser.add_argument('--compact-loss', action='store_true', default=False, help='use a compact form of loss function')
    parser.add_argument('--alpha', type=float, default=.001, help='adam-alpha')
    parser.add_argument('--betas', type=float, nargs=2, default=[.9, .999], help='adam-betas')
    parser.add_argument('--max-grad-norm', type=float, default=2., help='max grad norm allowed when clipping')
    parser.add_argument('--patience', type=int, default=10, help='number of epochs to wait when AUC does not improve')
    parser.add_argument('-r', '--repeat', type=int, default=1, help='times of repetition')
    parser.add_argument('-k', type=int, default=1, help='k-fold cross validation')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    k = args.k
    r = args.repeat
    start_time = time()

    for lambda_o in args.o:
        for lambda_w1 in args.w1:
            for lambda_w2 in args.w2:
                auc = np.zeros(r * k)
                rmse = np.zeros(r * k)
                mae = np.zeros(r * k)
                for j in range(r):
                    print('#%d:' % j)
                    for i in range(k):
                        print('group %d:' % i)
                        auc[j * k + i], rmse[j * k + i], mae[j * k + i] = train(args.problems, args.submissions, args.dir, j, i, lambda_o, lambda_w1, lambda_w2,
                                                                                hidden_size=args.hidden, dropout=args.dropout,
                                                                                lr=args.alpha, betas=args.betas, max_grad_norm=args.max_grad_norm,
                                                                                batch_size=args.batch, num_epochs=args.epochs, patience=args.patience,
                                                                                max_q=args.l, skills=args.skills, dump=args.dump,
                                                                                shuffle=args.shuffle, compact=args.compact_loss)
                        print('-' * 30)
                print()
                print('=' * 30)
                pattern = '{name}: {mean} (+/- {std})'
                print('o = %.1f, w1 = %.1f, w2 = %.1f' % (lambda_o, lambda_w1, lambda_w2))
                print(pattern.format(name='ROC AUC', mean=auc.mean(), std=auc.std()))
                print(pattern.format(name='RMSE', mean=rmse.mean(), std=rmse.std()))
                print(pattern.format(name='MAE', mean=mae.mean(), std=mae.std()))
                print('=' * 30)
                print()
                print()

    print('Elapsed time: ' + str(time() - start_time))


if __name__ == '__main__':
    main(sys.argv[1:])
