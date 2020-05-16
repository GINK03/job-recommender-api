import pickle
import gzip
import glob
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor
import joblib
import psutil
FILE = Path(__file__).name
TOP_DIR = Path(__file__).resolve().parent.parent

WORD_SIZE = 7264006

if "--transform" in sys.argv:
    """ 1000個づつ分割 """
    filenames = glob.glob(f"{TOP_DIR}/var/user_vectors/*")
    args = []
    STEP = 2000
    for i in range(0, len(filenames), STEP):
        args.append((i, filenames[i:i+STEP]))
    
    def load(arg):
        key, filenames = arg
        mtx = lil_matrix((STEP, 1000))
        usernames = []
        for idx, filename in enumerate(filenames):
            try:
                with open(filename, "rb") as fp:
                    vec = pickle.loads(gzip.decompress(fp.read()))
            except Exception as exc:
                tb_lineno = sys.exc_info()[2].tb_lineno
                print(f"[{FILE}] exc = {exc}, tb_lineno = {tb_lineno}", file=sys.stderr)
                continue
            for term_idx, weight in vec.items():
                mtx[idx, term_idx%1000] = weight
            usernames.append(Path(filename).name)
        data = (usernames, mtx.todense())
        print(len(usernames), mtx.todense().shape)
        if len(usernames) != mtx.todense().shape[0]:
            raise Exception("size not match!")
        with open(f"{TOP_DIR}/tmp/data/{key:09d}.pkl.gz", "wb") as fp:
            fp.write(gzip.compress(pickle.dumps(data)))

    with ProcessPoolExecutor(max_workers=psutil.cpu_count()) as exe:
        for _ in tqdm(exe.map(load, args), total=len(args), desc="transforming..."):
            _
