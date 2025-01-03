#!/bin/bash

#=================== 2 ===================
# P
# nohup python3 -u baseline/ea.py --variant P --path data/2m > logs/2m/P/ea.out &
# nohup python3 -u baseline/lp.py --variant P --path data/2m > logs/2m/P/lp.out &
# nohup python3 -u baseline/aco.py --variant P --path data/2m > logs/2m/P/aco.out &
# nohup python3 -u baseline/ils.py --variant P --path data/2m > logs/2m/P/ils.out &
# nohup python3 -u baseline/rl_hyb.py --variant P --cpkt ../cpkts/bestP_20_2.ckpt --path data/2m > logs/2m/P/rl_hyb.out &

# U
# nohup python3 -u baseline/ea.py --variant U --path data/2m > logs/2m/U/ea.out &
# nohup python3 -u baseline/lp.py --variant U --path data/2m > logs/2m/U/lp.out &
# nohup python3 -u baseline/aco.py --variant U --path data/2m > logs/2m/U/aco.out &
# nohup python3 -u baseline/ils.py --variant U --path data/2m > logs/2m/U/ils.out &
# nohup python3 -u baseline/rl_hyb.py --variant U --cpkt ../cpkts/bestU.ckpt --path data/2m > logs/2m/U/rl_hyb.out &



#=================== 3 ===================
# P
nohup python3 -u baseline/ea.py --variant P --path data/3m > logs/3m/P/ea.out &
# nohup python3 -u baseline/lp.py --variant P --path data/3m > logs/3m/P/lp.out &
# nohup python3 -u baseline/aco.py --variant P --path data/3m > logs/3m/P/aco.out &
# nohup python3 -u baseline/ils.py --variant P --path data/3m > logs/3m/P/ils.out &
# nohup python3 -u baseline/rl_hyb.py --variant P --cpkt ../cpkts/bestP_20_2.ckpt --path data/3m > logs/3m/P/rl_hyb.out &

# U
nohup python3 -u baseline/ea.py --variant U --path data/3m > logs/3m/U/ea.out &
# nohup python3 -u baseline/lp.py --variant U --path data/3m > logs/3m/U/lp.out &
# nohup python3 -u baseline/aco.py --variant U --path data/3m > logs/3m/U/aco.out &
# nohup python3 -u baseline/ils.py --variant U --path data/3m > logs/3m/U/ils.out &
# nohup python3 -u baseline/rl_hyb.py --variant U --cpkt ../cpkts/bestU.ckpt --path data/3m > logs/3m/U/rl_hyb.out &

# nohup python3 -u baseline/rl_hyb.py --variant U --cpkt ../cpkts/50P/epoch=008.ckpt --path data/5m > logs/5m/U/rl_hybnew.out &


# nohup python3 -u baseline/aco.py --variant U --path data/5m40 > logs/5m40/U/aco.out &
# nohup python3 -u baseline/ils.py --variant U --path data/5m40 > logs/5m40/U/ils.out &

# nohup python3 -u baseline/rl_hyb.py --variant U --cpkt ../cpkts/60U/epoch=049.ckpt --path data/5m60 > logs/5m60/U/rl_hyb.out &
# nohup python3 -u baseline/ils.py --variant U --path data/5m60 > logs/5m60/U/ils.out &
# nohup python3 -u baseline/ea.py --variant U --path data/5m60 > logs/5m60/U/ea.out &
# nohup python3 -u baseline/aco.py --variant U --path data/5m60 > logs/5m60/U/aco.out &
nohup python3 -u baseline/rl_hyb.py --variant U --cpkt ../cpkts/best60U.ckpt --path data/5m60 > logs/5m60/U/rl_hyb.out &

nohup python3 -u baseline/rl_hyb.py --variant U --cpkt ../cpkts/best60U.ckpt --path data/5m60 > logs/5m60/U/rl_hyb.out &

python3 -u baseline/rl_hyb.py --variant U --cpkt ../cpkts/bestU.ckpt --path data/5m