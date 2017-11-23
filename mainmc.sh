#!/bin/bash

#lr=(1 0.7 0.5 0.3 0.2 0.1 0.05 0.01)
#d=(1 0.9)
##n=(4 8 10 16)
#n=(8)
##iv=(-1 0 1 -100 100)
#iv=(0)
#
##parallel python3 mainmc.py --learning-rate {1} --discount {2} --behaviour-epsilon 0 --target-epsilon 0 --num-pos {3} --num-vel {3} --initial-value {4} ::: ${lr[*]} ::: ${d[*]} ::: ${n[*]} ::: ${iv[*]}
##parallel python3 mainmc.py --learning-rate 0.3 --discount {1} --behaviour-epsilon 0 --target-epsilon 0 --num-pos {2} --num-vel {2} --initial-value {3} --model rbf --results-dir ./results-rbf ::: ${d[*]} ::: ${n[*]} ::: ${iv[*]}
#parallel python3 mainmc.py --learning-rate {1} --discount {2} --behaviour-epsilon 0 --target-epsilon 0 --num-pos {3} --num-vel {3} --initial-value {4} --model rbf --results-dir ./results-rbf ::: ${lr[*]} ::: ${d[*]} ::: ${n[*]} ::: ${iv[*]}

python3 mainmc.py --model rbf --grid-search --results-dir ./results/rbf-gs
python3 mainmc.py --model rbf --best-params-from ./results/rbf-gs --results-dir ./results/rbf-best-mean --trials 500

python3 mainmc.py --model rbft --grid-search --results-dir ./results/rbft-gs
python3 mainmc.py --model rbft --best-params-from ./results/rbft-gs --results-dir ./results/rbft-best-mean --trials 500

python3 mainmc.py --model lstd-rbf --grid-search --results-dir ./results/lstd-rbf-gs
python3 mainmc.py --model lstd-rbf --best-params-from ./results/lstd-rbf-gs --results-dir ./results/lstd-rbf-best-mean --trials 500

python3 mainmc.py --graph --results-dirs ./results/rbf-best-mean ./results/rbft-best-mean ./results/lstd-rbf-best-mean
