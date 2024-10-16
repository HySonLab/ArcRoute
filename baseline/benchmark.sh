#!/bin/bash

#=================== 2 ===================
# P
# nohup python3 -u baseline/ea.py --variant P --path data/instances > logs/P/ea.out &
# nohup python3 -u baseline/lp.py --variant P --path data/instances > logs/P/lp.out &
nohup python3 -u baseline/aco.py --variant P --path data/instances > logs/P/aco.out &
# nohup python3 -u baseline/ils.py --variant P --path data/instances > logs/P/ils.out &
# nohup python3 -u baseline/rl_hyb.py --variant P --cpkt ../cpkts/bestP_20_2.ckpt --path data/instances > logs/P/rl_hyb.out &

# U
# nohup python3 -u baseline/ea.py --variant U --path data/instances > logs/U/ea.out &
# nohup python3 -u baseline/lp.py --variant U --path data/instances > logs/U/lp.out &
# nohup python3 -u baseline/aco.py --variant U --path data/instances > logs/U/aco.out &
# nohup python3 -u baseline/ils.py --variant U --path data/instances > logs/U/ils.out &
# nohup python3 -u baseline/rl_hyb.py --variant U --cpkt ../cpkts/bestU.ckpt --path data/instances > logs/U/rl_hyb.out &



#=================== 3 ===================
# P
# nohup python3 -u baseline/ea.py --variant P --path data/instances2 > logs/3/P/ea.out &
# nohup python3 -u baseline/lp.py --variant P --path data/instances2 > logs/3/P/lp.out &
# nohup python3 -u baseline/aco.py --variant P --path data/instances2 > logs/3/P/aco.out &
# nohup python3 -u baseline/ils.py --variant P --path data/instances2 > logs/3/P/ils.out &
# nohup python3 -u baseline/rl_hyb.py --variant P --cpkt ../cpkts/bestP_20_2.ckpt --path data/instances2 > logs/3/P/rl_hyb.out &

# U
# nohup python3 -u baseline/ea.py --variant U --path data/instances2 > logs/3/U/ea.out &
# nohup python3 -u baseline/lp.py --variant U --path data/instances2 > logs/3/U/lp.out &
# nohup python3 -u baseline/aco.py --variant U --path data/instances2 > logs/3/U/aco.out &
# nohup python3 -u baseline/ils.py --variant U --path data/instances2 > logs/3/U/ils.out &
# nohup python3 -u baseline/rl_hyb.py --variant U --cpkt ../cpkts/bestU.ckpt --path data/instances2 > logs/3/U/rl_hyb.out &
