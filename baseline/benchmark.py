import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baseline.meta_models import InsertCheapestHCARP, EAHCARP, ACOHCARP
from time import time
from glob import glob
import numpy as np

if __name__ == "__main__":
    np.random.seed(6868)
    files = glob('/usr/local/sra/ArcRoute/data/instances/*/*.npz')[:10]

    print('------------------')
    al = EAHCARP(n_population=200) # EA
    for f in files:
        al.import_instance(f)
        t1 = time()
        print(f,':::' , al(n_epoch=1, variant='P'),':::' , time() - t1)

    print('------------------')
    al = ACOHCARP(n_ant=100) # ACO
    for f in files:
        al.import_instance(f)
        t1 = time()
        print(f,':::', al(n_epoch=1, variant='P'),':::', time() - t1)

    print('------------------')
    al = InsertCheapestHCARP() # ACO
    for f in files:
        al.import_instance(f)
        t1 = time()
        print(f,':::', al(variant='P'),':::', time() - t1)