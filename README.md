# pyKT: PyTorch Implementation for Knowledge Tracing Models


This repo is part of the *ACM Assistant* project aimed to apply Knowledge Tracing (KT) to programming contests.

> Unfortunately, due to different platform/implementation/train-test-split, **NOT** all the implemented models have achieved comparable performance to that of the original implementations on the benchmark datasets.


## Environment
- Intel Core i9-9900K
- NVIDIA GeForce RTX 2070 SUPER
- Windows 10
- CUDA 10.1 + cuDNN 7.6
- Python 3.8
- PyTorch 1.6

> Other dependencies: Scikit-Learn, Matplotlib.


## Datasets

### Benchmark Datasets
- [ASSISTment 2009 (fixed)](/data/benchmarks/assist2009_updated)
- [ASSISTment 2015](/data/benchmarks/assist2015)
- [ASSISTment Challenge](/data/benchmarks/assistment_challenge)
- [STATICS](/data/benchmarks/STATICS)

> These datasets are from [here](https://github.com/ckyeungac/deep-knowledge-tracing-plus). The replication results can be found [here](/data/benchmarks).

### Online Judges
- [Codeforces (CF)](http://codeforces.com/)
- [HDU](http://acm.hdu.edu.cn/)
- [POJ](http://poj.org/)

*For HDU and POJ, there are no skill-level tags. I crawled the blogs on [CSDN](https://www.csdn.net/) to get very very rough tags for the problems (exercises).*

> Detailed information on the datasets can be found [here](/data).


## Models & Papers
- [Bayesian Knowledge Tracing (BKT)](/bkt.py)
  - *The implementation in this repo is merely a toy. For more standard, sophisticated implementation, refer to other repos.*
  - Corbett, A. T. & Anderson, J. R. (1995). Knowledge tracing: Modeling the acquisition of procedural knowledge. User Modeling and User-Adapted Interaction, 4, 253-278.
- [Performance Factors Analysis (PFA)](/pfa.py)
  - Pavlik, Philip I. & Cen, Hao & Koedinger, Kenneth R. (2009). Performance Factors Analysis -- A New Alternative to Knowledge Tracing. In Proceedings of the 2009 Conference on Artificial Intelligence in Education: Building Learning Systems That Care: From Knowledge Representation to Affective Modelling, 531–538.
- [Deep Knowledge Tracing (Plus) (DKT)](/dkt.py)
  - Yeung, Chun Kit & Yeung, Dit Yan (2018). Addressing Two Problems in Deep Knowledge Tracing via Prediction-Consistent Regularization. In Proceedings of the 5th ACM Conference on Learning @ Scale, 5:1-5:10.
- [Dynamic Key-Value Memory Network (DKVMN)](/dkvmn.py)
  - Jiani Zhang, Xingjian Shi, Irwin King, Dit-Yan Yeung (2017). Dynamic Key-Value Memory Networks for Knowledge Tracing. In Proceedings of the 26th International Conference on World Wide Web, 765-774.
- [Deep-IRT](/dkvmn.py)
  - Yeung, Chun Kit (2019). Deep-IRT: Make Deep Learning Based Knowledge Tracing Explainable Using Item Response Theory. In Proceedings of the 12th International Conference on Educational Data Mining, 683-686.
- [Self-Attentive Knowledge Tracing (SAKT)](/sakt.py)
  - Pandey, Shalini & Karypis, George (2019). A Self-Attentive model for Knowledge Tracing. In Proceedings of the 12th International Conference on Educational Data Mining, 384-389.


## Results & Parameters

|Model|CF|HDU|POJ|
|:-:|:-:|:-:|:-:|
|PFA|0.657|0.663|0.679|
|DKT|0.742|0.704|0.721|
|DKVMN|0.749|0.728|0.751|
|Deep-IRT|0.748|0.730|0.754|
|SAKT|0.756|0.761|0.773|

> For the time being, the parameters are **NOT** carefully tuned.

PFA Parameters:
```
python .\pfa.py -p .\data\CF\problems.json -s .\data\CF\user_submissions.json -k 5

python .\pfa.py -p .\data\HDU\problems.json -s .\data\HDU\user_submissions.json -k 5

python .\pfa.py -p .\data\POJ\problems.json -s .\data\POJ\user_submissions.json -k 5
```

DKT Parameters:
```
python .\dkt.py -p .\data\CF\problems.json -s .\data\CF\user_submissions.json -D .\models\ -H 512 --dropout 0.5 --shuffle --compact-loss -k 5

python .\dkt.py -p .\data\HDU\problems.json -s .\data\HDU\user_submissions.json -D .\models\ -H 512 --dropout 0.5 --shuffle --compact-loss -k 5

python .\dkt.py -p .\data\POJ\problems.json -s .\data\POJ\user_submissions.json -D .\models\ -H 512 --dropout 0.5 --shuffle --compact-loss -k 5
```

DKVMN Parameters:
```
python .\dkvmn.py -p .\data\CF\problems.json -s .\data\CF\user_submissions.json -D .\models\ -d_k 128 -d_v 256 -H 1024 -n 50 --shuffle --dropout 0.5 -k 5

python .\dkvmn.py -p .\data\HDU\problems.json -s .\data\HDU\user_submissions.json -D .\models\ -d_k 128 -d_v 256 -H 1024 -n 50 --shuffle --dropout 0.5 -k 5

python .\dkvmn.py -p .\data\POJ\problems.json -s .\data\POJ\user_submissions.json -D .\models\ -d_k 128 -d_v 256 -H 1024 -n 50 --shuffle --dropout 0.5 -k 5
```

Deep-IRT Parameters:
```
python .\dkvmn.py -p .\data\CF\problems.json -s .\data\CF\user_submissions.json -D .\models\ -d_k 128 -d_v 256 -H 512 -n 20 --deep-irt --shuffle --dropout 0.5 -k 5

python .\dkvmn.py -p .\data\HDU\problems.json -s .\data\HDU\user_submissions.json -D .\models\ -d_k 128 -d_v 256 -H 512 -n 20 --deep-irt --shuffle --dropout 0.5 -k 5

python .\dkvmn.py -p .\data\POJ\problems.json -s .\data\POJ\user_submissions.json -D .\models\ -d_k 128 -d_v 256 -H 512 -n 20 --deep-irt --shuffle --dropout 0.5 -k 5
```

SAKT Parameters:
```
python .\sakt.py -p .\data\CF\problems.json -s .\data\CF\user_submissions.json -D .\models\ -e 100 -d 128 -H 512 --heads 8 -n 5 --dropout 0.2 -k 5

python .\sakt.py -p .\data\HDU\problems.json -s .\data\HDU\user_submissions.json -D .\models\ -e 100 -d 128 -H 512 --heads 8 -n 5 --dropout 0.2 -k 5

python .\sakt.py -p .\data\POJ\problems.json -s .\data\POJ\user_submissions.json -D .\models\ -e 100 -d 128 -H 512 --heads 8 -n 5 --dropout 0.2 -k 5
```
