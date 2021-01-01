# Dataset Description

Each dataset contains two json files: `problems.json` and `user_submissions.json`.

- `problems.json` contains problem information in the following JSON format:
  ```
  [
      {"id": 0, "tags": [10, 11, 21]},
      {"id": 1, "tags": [5, 7, 22]},
      {"id": 2, "tags": [2, 6, 12, 16, 18, 23]},
      {"id": 3, "tags": [2, 4, 5, 9, 10, 11, 18]},
      {"id": 4, "tags": [0, 3, 6, 12]},
      ...
  ]
  ```
  - `id` is the problem id starting from 0.
  - `tags` is a list of tag ids also starting from 0.

- `user_submissions.json` contains user submissions in the following JSON format:
  ```
  [
      {
          "group": 0,
          "submissions":
          [
              {"problem": 995, "verdict": 1},
              {"problem": 994, "verdict": 0},
              {"problem": 994, "verdict": 1},
              {"problem": 993, "verdict": 1},
              {"problem": 992, "verdict": 1},
              ...
          ]
      },
      {
          "group": 1,
          "submissions":
          [
              {"problem": 5679, "verdict": 0},
              {"problem": 5679, "verdict": 1},
              {"problem": 5533, "verdict": 0},
              {"problem": 5533, "verdict": 1},
              {"problem": 5532, "verdict": 1},
              ...
          ]
      },
      ...
  ]
  ```
  - `group` is used to distinguish the test set from the training set.
    > You may notice that there is a parameter `k` in the source code. For [`CF`](/data/CF), [`HDU`](/data/HDU) and [`POJ`](/data/POJ), `group` is assigned to 0, 1, 2, 3, or 4, so that you may perform 5-fold cross validation by setting `k` to 5. For the benchmark [datasets](/data/benchmarks), `group` is set to 0 for the test set and 1 for the training set.
  - `submissions` is a list of submissions in (ascending) time order.
  - `problem` is the problem id of a submission.
  - `verdict` is the result of a submission. `1` is for `OK` (correct) and `0` is for `FAILED` (incorrect).
