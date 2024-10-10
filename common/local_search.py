import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.nb_utils import gen_tours_batch
from common.intra import intraP, intraU
from common.inter import interP, interU
from common.ops import run_parallel, convert_vars_np


def local_search(vars, variant, actions=None, tours_batch=None, is_train=True):
    if tours_batch is None:
        tours_batch = gen_tours_batch(actions)
        
    if not isinstance(vars, dict):
        vars = convert_vars_np(vars)

    bs = len(tours_batch)
    if is_train:
        tours_batch = run_parallel(intraU if variant=='U' else intraP, [1]*bs, 
                                vars['adj'], 
                                vars['service_time'], 
                                vars['clss'], 
                                tours_batch)
    else:
        for _ in range(1):
            for k in [1,2,3]:
                # tours_batch = run_parallel(intraU if variant=='U' else intraP, [k]*bs, 
                #                         vars['adj'], 
                #                         vars['service_time'], 
                #                         vars['clss'], 
                #                         tours_batch)
                tours_batch = run_parallel(interU if variant=='U' else interP, [k]*bs, 
                                        vars['adj'], 
                                        vars['service_time'], 
                                        vars['clss'],
                                        vars['demand'],
                                        tours_batch)

    return tours_batch

