# Replication of the Results


*If you have any idea on how to improve, please let me know.*


> I do not know why SAKT has such an astonishing performance. Perhaps there is something wrong with my source code.


|Model|ASSIST2009|ASSIST2015|STATICS|
|:-:|:-:|:-:|:-:|
|PFA|0.715 (*0.597*)|0.686 (*0.529*)|0.689 (*0.650*)|
|DKT|0.823 (*0.821*)|0.708 (*0.736*)|0.831 (*0.827*)|
|DKVMN|0.815 (*0.816*)|0.707 (*0.729*)|0.825 (*0.832*)|
|Deep-IRT|0.812 (*0.817*)|0.705 (*0.729*)|0.825 (*0.831*)|
|SAKT|**0.910** (*0.848*)|**0.918** (*0.854*)|**0.834** (*0.853*)| 

*Italics* are (nearly) the best reported results in the papers.


> If you are getting "RuntimeError: CUDA error: unspecified launch failure" on a random basis when using DKT (LSTM), it happens to me as well. It is a known, open [issue](https://github.com/pytorch/pytorch/issues/27837) in PyTorch and **NOT** related to my source code.


PFA Parameters:
```
python .\pfa.py -p .\data\benchmarks\assist2009_updated\problems.json -s .\data\benchmarks\assist2009_updated\user_submissions.json

python .\pfa.py -p .\data\benchmarks\assist2015\problems.json -s .\data\benchmarks\assist2015\user_submissions.json

python .\pfa.py -p .\data\benchmarks\STATICS\problems.json -s .\data\benchmarks\STATICS\user_submissions.json
```

DKT Parameters:
```
python .\dkt.py -p .\data\benchmarks\assist2009_updated\problems.json -s .\data\benchmarks\assist2009_updated\user_submissions.json -D .\models\ -e 100 -H 200 --dropout 0.5 --alpha 0.01 --max-grad-norm 5.0 -r 5 --shuffle --compact-loss

python .\dkt.py -p .\data\benchmarks\assist2015\problems.json -s .\data\benchmarks\assist2015\user_submissions.json -D .\models\ -e 100 -H 200 --dropout 0.5 --alpha 0.01 --max-grad-norm 5.0 -r 5 --shuffle --compact-loss

python .\dkt.py -p .\data\benchmarks\STATICS\problems.json -s .\data\benchmarks\STATICS\user_submissions.json -D .\models\ -e 100 -H 100 --dropout 0.5 --alpha 0.003 --max-grad-norm 10.0 -r 5 --shuffle --compact-loss
```

DKVMN Parameters:
```
python .\dkvmn.py -p .\data\benchmarks\assist2009_updated\problems.json -s .\data\benchmarks\assist2009_updated\user_submissions.json -D .\models\ -e 100 -d_k 100 -d_v 200 -n 10 -H 512 --shuffle --dropout 0.5 --alpha 0.003 --max-grad-norm 10.0 -r 5

python .\dkvmn.py -p .\data\benchmarks\assist2015\problems.json -s .\data\benchmarks\assist2015\user_submissions.json -D .\models\ -e 100 -d_k 50 -d_v 100 -H 200 -n 100 --shuffle --dropout 0.5 --alpha 0.003 --max-grad-norm 10.0 -r 5

python .\dkvmn.py -p .\data\benchmarks\STATICS\problems.json -s .\data\benchmarks\STATICS\user_submissions.json -D .\models\ -e 100 -d_k 200 -d_v 200 -n 5 -H 200 --shuffle --dropout 0.5 --alpha 0.003 --max-grad-norm 5.0 -r 5
```

Deep-IRT Parameters:
```
python .\dkvmn.py -p .\data\benchmarks\assist2009_updated\problems.json -s .\data\benchmarks\assist2009_updated\user_submissions.json -D .\models\ -e 100 -d_k 100 -d_v 200 -n 10 -H 200 --deep-irt --shuffle --dropout 0.5 --alpha 0.003 --max-grad-norm 10.0 -r 5

python .\dkvmn.py -p .\data\benchmarks\assist2015\problems.json -s .\data\benchmarks\assist2015\user_submissions.json -D .\models\ -e 100 -d_k 50 -d_v 100 -H 200 -n 100 --deep-irt --shuffle --dropout 0.5 --alpha 0.003 --max-grad-norm 10.0 -r 5

python .\dkvmn.py -p .\data\benchmarks\STATICS\problems.json -s .\data\benchmarks\STATICS\user_submissions.json -D .\models\ -e 100 -d_k 200 -d_v 200 -n 5 -H 200 --deep-irt --shuffle --dropout 0.5 --alpha 0.003 --max-grad-norm 5.0 -r 5
```

SAKT Parameters:
```
python .\sakt.py -p .\data\benchmarks\assist2009_updated\problems.json -s .\data\benchmarks\assist2009_updated\user_submissions.json -D .\models\ -e 100 -b 32 -d 128 -H 256 --heads 8 -n 5 --dropout 0.2 -r 5

python .\sakt.py -p .\data\benchmarks\assist2015\problems.json -s .\data\benchmarks\assist2015\user_submissions.json -D .\models\ -e 100 -d 128 -H 512 --heads 8 -n 3 --dropout 0.2 -r 5

python .\sakt.py -p .\data\benchmarks\STATICS\problems.json -s .\data\benchmarks\STATICS\user_submissions.json -D .\models\ -e 100 -b 32 -d 64 -H 128 --heads 16 -n 8 --dropout 0.2 -r 5
```
