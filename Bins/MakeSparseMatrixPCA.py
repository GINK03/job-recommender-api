import pickle
import gzip
import glob
from scipy.sparse import lil_matrix
from sklearn.decomposition import MiniBatchSparsePCA
# import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor
import joblib
import pandas as pd
from os import environ as E
import psutil
import time

HOME = Path.home()
FILE = Path(__file__).name
TOP_DIR = Path(__file__).resolve().parent.parent

idf = pd.read_csv(f'{TOP_DIR}/var/doc_freq.csv')
WORD_SIZE = len(idf)

if "--create_transformer" in sys.argv:
    SAMPLE_SIZE = 10000
    print(f"total word size is = {WORD_SIZE}")
    start_time = time.time() 
    mtx = lil_matrix((SAMPLE_SIZE, WORD_SIZE))
    def load(arg):
        idx, filename = arg
        try:
            with open(filename, "rb") as fp:
                vec = pickle.loads(gzip.decompress(fp.read()))
            return (idx, vec)
        except Exception as exc:
            print(exc, idx, filename)
            return None
    args = []
    for idx, filename in tqdm(enumerate(glob.glob(f"{TOP_DIR}/var/user_vectors/*")[:SAMPLE_SIZE]), desc="load example users..."):
        args.append((idx, filename))
    with ProcessPoolExecutor(max_workers=psutil.cpu_count()) as exe:
        for ret in tqdm(exe.map(load, args), total=len(args), desc="load example users..."):
            if ret is None:
                continue
            idx, vec = ret
            for term_idx, weight in vec.items():
                mtx[idx, term_idx] = weight

    print(f"[{FILE}] start to train TruncatedSVD...")
    transformer = MiniBatchSparsePCA(n_components=500, batch_size=100, random_state=0)
    transformer.fit(mtx.todense())
    elapsed_time = time.time() - start_time
    print(f"[{FILE}] elapsed_time = {elapsed_time}")
    print(f"[{FILE}] start to transform matrix...")
    X_transformed = transformer.transform(mtx[:5000])
    print(X_transformed)
    print(X_transformed.shape)
    print(type(X_transformed))
    joblib.dump(transformer, f"{TOP_DIR}/var/transformer.joblib")

if "--transform" in sys.argv:
    transformer = joblib.load(f"{TOP_DIR}/var/transformer.joblib")
    """ 1000個づつ分割 """
    filenames = glob.glob(f"{HOME}/var/user_vectors/*")
    args = []
    STEP = 2000
    for i in range(0, len(filenames), STEP):
        args.append((i, filenames[i:i+STEP]))
   
    Path(f"{TOP_DIR}/tmp/data_svd").mkdir(exist_ok=True, parents=True)
    def load(arg):
        key, filenames = arg
        mtx = lil_matrix((STEP, WORD_SIZE))
        usernames = []
        for idx, filename in enumerate(filenames):
            usernames.append(Path(filename).name)
            try:
                with open(filename, "rb") as fp:
                    vec = pickle.loads(gzip.decompress(fp.read()))
            except Exception as exc:
                tb_lineno = sys.exc_info()[2].tb_lineno
                print(f"[{FILE}] exc = {exc}, tb_lineno = {tb_lineno}", file=sys.stderr)
                continue
            for term_idx, weight in vec.items():
                mtx[idx, term_idx] = weight
        X_transformed = transformer.transform(mtx)
        data = (usernames, X_transformed)
        print(len(usernames), X_transformed.shape)
        if len(usernames) != X_transformed.shape[0]:
            raise Exception("size not match!")
        with open(f"{TOP_DIR}/tmp/data_svd/{key:09d}.pkl.gz", "wb") as fp:
            fp.write(gzip.compress(pickle.dumps(data)))

    with ProcessPoolExecutor(max_workers=psutil.cpu_count()) as exe:
        for _ in tqdm(exe.map(load, args), total=len(args), desc="transforming..."):
            _
