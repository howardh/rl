#!/bin/bash

lr=(1 0.7 0.5 0.3 0.2 0.1 0.05 0.01)
d=(1 0.9)
#n=(4 8 10 16)
n=(8)
#iv=(-1 0 1 -100 100)
iv=(0)

#parallel python3 mainmc.py --learning-rate {1} --discount {2} --behaviour-epsilon 0 --target-epsilon 0 --num-pos {3} --num-vel {3} --initial-value {4} ::: ${lr[*]} ::: ${d[*]} ::: ${n[*]} ::: ${iv[*]}
#parallel python3 mainmc.py --learning-rate 0.3 --discount {1} --behaviour-epsilon 0 --target-epsilon 0 --num-pos {2} --num-vel {2} --initial-value {3} --model rbf --results-dir ./results-rbf ::: ${d[*]} ::: ${n[*]} ::: ${iv[*]}
parallel python3 mainmc.py --learning-rate {1} --discount {2} --behaviour-epsilon 0 --target-epsilon 0 --num-pos {3} --num-vel {3} --initial-value {4} --model rbf --results-dir ./results-rbf ::: ${lr[*]} ::: ${d[*]} ::: ${n[*]} ::: ${iv[*]}
