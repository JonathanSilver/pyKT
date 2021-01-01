import numpy as np
from math import log
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, classification_report
from math import sqrt
import json
from pprint import pprint
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--problems', type=str, help='file path to problems.json')
parser.add_argument('-s', '--submissions', type=str, help='file path to user_submissions.json')
parser.add_argument('-k', type=int, default=1, help='k-fold cross validation')
args = parser.parse_args()

with open(args.problems, 'r') as file:
    problems = json.load(file)
problem_id_2_tag_ids = {problem['id']: problem['tags'] for problem in problems}
with open(args.submissions, 'r') as file:
    user_submissions = json.load(file)
max_skill = max([max(problem['tags']) for problem in problems if len(problem['tags']) > 0]) + 1
print('max_skill:', max_skill)


def read_data(training, group, expand_tags=False):
    x = []
    y = []

    for user_data in user_submissions:
        user_group = user_data['group']
        if training and user_group == group \
                or not training and user_group != group:
            continue
        submissions = user_data['submissions']
        user_success = {}
        user_fail = {}
        for sub in submissions:
            tags = problem_id_2_tag_ids[sub['problem']]
            if not expand_tags:
                y.append(sub['verdict'])
                x.append([0] * 3 * max_skill)
                for tag in tags:
                    s = user_success.get(tag, 1)
                    f = user_fail.get(tag, 1)
                    x[-1][tag * 3 + 0] = 1
                    x[-1][tag * 3 + 1] = log(s)
                    x[-1][tag * 3 + 2] = log(f)
                    if sub['verdict'] == 1:
                        user_success[tag] = s + 1
                    else:
                        user_fail[tag] = f + 1
            else:
                for tag in tags:
                    s = user_success.get(tag, 1)
                    f = user_fail.get(tag, 1)
                    x.append([0] * 3 * max_skill)
                    x[-1][tag * 3 + 0] = 1
                    x[-1][tag * 3 + 1] = log(s)
                    x[-1][tag * 3 + 2] = log(f)
                    if sub['verdict'] == 1:
                        y.append(1)
                        user_success[tag] = s + 1
                    else:
                        y.append(0)
                        user_fail[tag] = f + 1
    return x, y


def train(group):
    model = LogisticRegression()
    x, y = read_data(training=True, group=group, expand_tags=False)
    print('Fitting')
    model.fit(x, y)
    x, y = read_data(training=False, group=group, expand_tags=False)
    print('Predicting')
    pred = model.predict_proba(x)[:, 1]
    auc = roc_auc_score(y, pred)
    rmse = sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    print('ROC AUC: {}'.format(auc))
    print('RMSE: {}'.format(rmse))
    print('MAE: {}'.format(mae))
    # res = np.zeros(pred.shape[0])
    # res[pred >= 0.5] = 1
    # print(classification_report(y, res))
    return auc, rmse, mae


def main():
    k = args.k
    auc = np.zeros(k)
    rmse = np.zeros(k)
    mae = np.zeros(k)
    for i in range(k):
        print('group: %d' % i)
        auc[i], rmse[i], mae[i] = train(i)
        print('-' * 20)
    print('ROC AUC: {} (+/- {})'.format(auc.mean(), auc.std()))
    print('RMSE: {} (+/- {})'.format(rmse.mean(), rmse.std()))
    print('MAE: {} (+/- {})'.format(mae.mean(), mae.std()))


if __name__ == '__main__':
    main()
