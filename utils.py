import numpy as np
import pandas as pd

import json

from pprint import pprint


def show_info(problems_file, user_submissions_file, group=-1):
    with open(user_submissions_file, 'r') as file:
        dataset = json.load(file)
    with open(problems_file, 'r') as file:
        problem2id = json.load(file)
        problem_id_2_tag_ids = {
            problem['id']: set(problem['tags'])
            for problem in problem2id
        }
    num_students = 0
    num_submissions = 0
    num_ok_submissions = 0
    num_failed_submissions = 0
    tags = set()
    problems = set()
    max_tag_id = -1
    for user in dataset:
        user_group = user['group']
        if group != -1 and user_group != group:
            continue
        num_students += 1
        submissions = user['submissions']
        num_submissions += len(submissions)
        for submission in submissions:
            tags |= problem_id_2_tag_ids[submission['problem']]
            if len(problem_id_2_tag_ids[submission['problem']]):
                max_tag_id = max(max_tag_id, max(list(problem_id_2_tag_ids[submission['problem']])))
            problems.add(submission['problem'])
            if submission['verdict'] == 1:
                num_ok_submissions += 1
            else:
                num_failed_submissions += 1
    print('Number of Students: %d' % num_students)
    print('Number of Problems: %d' % len(problems))
    print('Number of Tags: %d' % len(tags))
    print('Number of OK Submissions: %d' % num_ok_submissions)
    print('Number of FAILED Submissions: %d' % num_failed_submissions)
    print('Total Submissions: %d' % num_submissions)
    print(list(sorted(list(tags))))
    print(max_tag_id)


def read_csv(file_name, problems, submissions, group):
    with open(file_name) as file:
        while True:
            n = file.readline()
            if n == '':
                break
            n = int(n)
            q = file.readline().strip().split(',')
            r = file.readline().strip().split(',')
            assert n == len(q) and n == len(r)
            res = []
            for x, y in zip(q, r):
                problems.add(int(x))
                res.append({
                    'problem': int(x),
                    'verdict': int(y)
                })
            submissions.append({'submissions': res, 'group': group})


def convert_csv_2_json(root_dir, dataset_name, has_skills=False):
    problems = set()
    submissions = []
    read_csv(root_dir + ('csv/%s_test.csv' % dataset_name), problems, submissions, 0)
    read_csv(root_dir + ('csv/%s_train.csv' % dataset_name), problems, submissions, 1)
    problems = {p: i for i, p in enumerate(problems)}
    submissions = [{
        'submissions': [{
            'problem': problems[s['problem']],
            'verdict': s['verdict']
        } for s in d['submissions']],
        'group': d['group']
    } for d in submissions]
    if has_skills:
        qid_2_sid = pd.read_csv(root_dir + ('csv/%s_qid_sid_sname' % dataset_name), sep='\t')
        qid_2_sid = qid_2_sid.iloc[:, :-1]
        qid_2_sid = np.array(qid_2_sid)
        qid_2_sid = {int(qid): int(sid) for qid, sid in qid_2_sid}
        tags = set(qid_2_sid.values())
        tags = {t: i for i, t in enumerate(tags)}
    json.dump([{
        'id': p,
        'tags': [tags[qid_2_sid[k]] if has_skills else p]
    } for k, p in problems.items()], open(root_dir + 'problems.json', 'w'))
    json.dump(submissions, open(root_dir + 'user_submissions.json', 'w'))


if __name__ == '__main__':
    dataset_name = 'statics'
    root_dir = './data/benchmarks/%s/' % dataset_name
    # convert_csv_2_json(root_dir, dataset_name, has_skills=True)
    # with open(root_dir + 'problems.json') as file:
    #     pprint(json.load(file)[-10:])
    # with open(root_dir + 'user_submissions.json') as file:
    #     pprint(json.load(file)[0])

    print('Overall Statistics')
    show_info(root_dir + 'problems.json', root_dir + 'user_submissions.json')

    # print('=' * 30)
    # for i in range(5):
    #     print('-' * 30)
    #     print('Statistics for group %d' % i)
    #     show_info(root_dir + 'problems.json', root_dir + 'user_submissions.json', i)
    #     print('-' * 30)
