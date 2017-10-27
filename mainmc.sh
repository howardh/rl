#!/bin/bash

lr=(1 0.5 0.2)
d=(1 0.9)
n=(4 8 10 16)
iv=(-1 0 1 -100 100)

#parallel python3 mainmc.py --learning-rate {1} --discount {2} --behaviour-epsilon 0 --target-epsilon 0 --num-pos {3} --num-vel {3} --initial-value {4} ::: ${lr[*]} ::: ${d[*]} ::: ${n[*]} ::: ${iv[*]}
parallel python3 mainmc.py --learning-rate 0.3 --discount {1} --behaviour-epsilon 0 --target-epsilon 0 --num-pos {2} --num-vel {2} --initial-value {3} ::: ${d[*]} ::: ${n[*]} ::: ${iv[*]}
