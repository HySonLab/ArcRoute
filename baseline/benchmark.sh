#!/bin/bash

# P
nohup python3 -u baseline/ea.py --variant P > logs/P/ea.out &
# nohup python3 -u baseline/lp.py > logs/P/lp.out &
nohup python3 -u baseline/aco.py --variant P > logs/P/aco.out &
nohup python3 -u baseline/ils.py --variant P > logs/P/ils.out &
nohup python3 -u baseline/rl_hyb.py --variant P --cpkt /usr/local/rsa/cpkts/bestP_20_2.ckpt > logs/P/rl_hyb.out &

# U
nohup python3 -u baseline/ea.py --variant U > logs/U/ea.out &
# nohup python3 -u baseline/lp.py --variant U > logs/U/lp.out &
nohup python3 -u baseline/aco.py --variant U > logs/U/aco.out &
nohup python3 -u baseline/ils.py --variant U > logs/U/ils.out &
nohup python3 -u baseline/rl_hyb.py --variant U --cpkt /usr/local/rsa/cpkts/bestU.ckpt > logs/U/rl_hyb.out &

