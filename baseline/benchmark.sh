#!/bin/bash

nohup python3 -u baseline/ea.py > logs/ea.out &
nohup python3 -u baseline/lp.py > logs/lp.out &
nohup python3 -u baseline/aco.py > logs/aco.out &
nohup python3 -u baseline/ils.py > logs/ils.out &
# nohup python3 -u baseline/rl_hyb.py > logs/rl_hyb.out &