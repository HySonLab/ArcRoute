import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baseline.meta_models import InsertCheapestHCARP
from time import time
from glob import glob
import numpy as np

if __name__ == "__main__":
    np.random.seed(6868)
    files = glob('/usr/local/sra/ArcRoute/data/instances/*/*.npz')

    al = InsertCheapestHCARP() # ILS
    for f in files:
        al.import_instance(f)
        t1 = time()
        print(f,':::', al(variant='P'),':::', time() - t1)