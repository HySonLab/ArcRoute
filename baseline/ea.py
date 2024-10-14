import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baseline.meta_models import EAHCARP
from time import time
from glob import glob
import numpy as np

if __name__ == "__main__":
    np.random.seed(6868)
    files = glob('/usr/local/rsa/ArcRoute/data/instances/*/*.npz')

    al = EAHCARP(n_population=100) # EA
    for f in files:
        al.import_instance(f)
        t1 = time()
        print(f,':::' , al(n_epoch=200, variant='P'),':::' , time() - t1)