import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.nb_utils import gen_tours_batch
from common.intra import intraP, intraU
from common.inter import interP, interU
from common.ops import run_parallel, convert_vars_np
from numpy.random import random

def ls(vars, variant, actions=None, tours_batch=None):
    if tours_batch is None:
        tours_batch = gen_tours_batch(actions)

    for _ in range(3):
        for k in [1,2,3]:
            # if random() > 0.5:
            tours_batch = run_parallel(intraU if variant=='U' else intraP, 
                                    tours_batch,
                                    adj=vars['adj'], 
                                    service=vars['service_time'], 
                                    clss=vars['clss'], 
                                    k = k)
            # if random() > 0.5:
            tours_batch = run_parallel(interU if variant=='U' else interP, 
                                    tours_batch,
                                    adj=vars['adj'], 
                                    service=vars['service_time'], 
                                    clss=vars['clss'],
                                    demand=vars['demand'],
                                    k = k)

    return tours_batch

def lsRL(vars, variant, actions=None, tours_batch=None, is_train=True):
    if tours_batch is None:
        tours_batch = gen_tours_batch(actions)
        
    if not isinstance(vars, dict):
        vars = convert_vars_np(vars)

    if is_train:
        for _ in range(2):
            tours_batch = run_parallel(intraU if variant=='U' else intraP, 
                                    tours_batch,
                                    vars['adj'], 
                                    vars['service_time'], 
                                    vars['clss'], 
                                    k = 1)
            tours_batch = run_parallel(interU if variant=='U' else interP, 
                                    tours_batch,
                                    vars['adj'], 
                                    vars['service_time'], 
                                    vars['clss'],
                                    vars['demand'],
                                    k = 1)
        
    else:
        for _ in range(3):
            for k in [1,2,3]:
                tours_batch = run_parallel(intraU if variant=='U' else intraP, 
                                        tours_batch,
                                        vars['adj'], 
                                        vars['service_time'], 
                                        vars['clss'], 
                                        k = k)
                tours_batch = run_parallel(interU if variant=='U' else interP, 
                                        tours_batch,
                                        vars['adj'], 
                                        vars['service_time'], 
                                        vars['clss'],
                                        vars['demand'],
                                        k = k)

    return tours_batch

