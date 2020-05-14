import pickle
import gzip
import glob
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor
import joblib
FILE = Path(__file__).name
TOP_DIR = Path(__file__).resolve().parent.parent

WORD_SIZE = 7264006

if "--create_transformer" in sys.argv:
    mtx = lil_matrix((10000, WORD_SIZE))
    for idx, filename in tqdm(enumerate(glob.glob(f"{TOP_DIR}/var/user_vectors/*")[:10000]), desc="load example users..."):
        with open(filename, "rb") as fp:
            vec = pickle.loads(gzip.decompress(fp.read()))
        for term_idx, weight in vec.items():
            mtx[idx, term_idx] = weight

    print(f"[{FILE}] start to train TruncatedSVD...")
    transformer = TruncatedSVD(n_components=100, random_state=0)
    transformer.fit(mtx)
    print(f"[{FILE}] start to transform matrix...")
    X_transformed = transformer.transform(mtx)
    print(X_transformed)
    print(X_transformed.shape)
    print(type(X_transformed))
    # with open("transformer.pkl", "wb") as fp:
    #    fp.write(pickle.dumps(transformer))
    joblib.dump(transformer, f"{TOP_DIR}/var/transformer.joblib")

if "--transform" in sys.argv:
    transformer = joblib.load(f"{TOP_DIR}/var/transformer.joblib")
    """ 1000個づつ分割 """
    filenames = glob.glob(f"{TOP_DIR}/var/user_vectors/*")
    args = []
    STEP = 5000
    for i in range(0, len(filenames), STEP):
        args.append((i, filenames[i:i+STEP]))
    
    def load(arg):
        key, filenames = arg
        mtx = lil_matrix((STEP, WORD_SIZE))
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
                mtx[idx, term_idx] = weight
            usernames.append(Path(filename).name)
        X_transformed = transformer.transform(mtx)
        data = (usernames, X_transformed)
        print(len(usernames), X_transformed.shape)
        if len(usernames) != X_transformed.shape[0]:
            raise Exception("size not match!")
        with open(f"{TOP_DIR}/tmp/data/{key:09d}.pkl.gz", "wb") as fp:
            fp.write(gzip.compress(pickle.dumps(data)))

    with ProcessPoolExecutor(max_workers=16) as exe:
        for _ in tqdm(exe.map(load, args), total=len(args), desc="transforming..."):
            _
