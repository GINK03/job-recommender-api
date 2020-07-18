from scipy.sparse import vstack
import pickle
import gzip
import glob
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD
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
import random
import os
from loguru import logger

logger.add(sys.stdout, format="{filename} {name} {time} {level} {message}", filter="my_module", level="DEBUG")

HOME = Path.home()
FILE = Path(__file__).name
TOP_DIR = Path(__file__).resolve().parent.parent

idf = pd.read_csv(f'{TOP_DIR}/var/doc_freq.csv')
idf = idf[:1000000]
WORD_SIZE = len(idf)
BATCH_SIZE = 2000


def _make(arg):
    i, input_user_files = arg
    mtx = lil_matrix((BATCH_SIZE, WORD_SIZE))
    users = [None]*BATCH_SIZE
    for idx, filename in tqdm(enumerate(input_user_files), desc=f"load example users, pid = {os.getpid()}..."):
        users[idx] = Path(filename).name
        with open(filename, "rb") as fp:
            vec = pickle.loads(gzip.decompress(fp.read()))
        for term_idx, weight in vec.items():
            if term_idx >= WORD_SIZE:
                continue
            mtx[idx, term_idx] = weight
    with open(f"{TOP_DIR}/var/mtx_{i:016d}.pkl", "wb") as fp:
        pickle.dump((users,mtx), fp)
    del mtx


def make_lil_cache():
    input_user_files = glob.glob(f"{TOP_DIR}/var/user_vectors/*")

    i_all = [(i, input_user_files[i:i+BATCH_SIZE]) for i in range(0, len(input_user_files), BATCH_SIZE)]
    with ProcessPoolExecutor(max_workers=psutil.cpu_count()) as exe:
        exe.map(_make, i_all)


if "--create_cache" in sys.argv:
    make_lil_cache()

if "--create_transformer" in sys.argv:
    logger.info(f"total word size is = {WORD_SIZE}")
    start_time = time.time()

    transformer = TruncatedSVD(n_components=500, n_iter=5, random_state=0)
    users, mtx = pickle.load(open("../var/mtx_0000000000000000.pkl", "rb"))
    for mtx_filename in glob.glob("../var/mtx_*.pkl")[:120]:
        if mtx_filename == "../var/mtx_0000000000000000.pkl":
            continue
        users, sub_mtx = pickle.load(open(mtx_filename, "rb"))
        mtx = vstack([mtx, sub_mtx])
        print(mtx.shape)
    mtx = mtx.tolil()
    logger.debug(f"start to train TruncatedSVD...")
    transformer.fit(mtx)
    elapsed_time = time.time() - start_time
    logger.debug(f"elapsed_time = {elapsed_time}")

    logger.debug(f"start to transform matrix...")
    X_transformed = transformer.transform(mtx[:5000])
    joblib.dump(transformer, f"{TOP_DIR}/var/transformer.joblib")

if "--transform" in sys.argv:
    transformer = joblib.load(f"{TOP_DIR}/var/transformer.joblib")
    """ 20000個づつ分割 """
    filenames = glob.glob(f"{TOP_DIR}/var/user_vectors/*")
    args = []
    STEP = 2000
    for i in range(0, len(filenames), STEP):
        args.append((i, filenames[i:i+STEP]))

    Path(f"{TOP_DIR}/var/data_svd").mkdir(exist_ok=True, parents=True)

    def load(arg):
        try:
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
                    logger.debug(f"[{FILE}] exc = {exc}, tb_lineno = {tb_lineno}")
                    continue
                for term_idx, weight in vec.items():
                    if term_idx >= WORD_SIZE:
                        continue
                    mtx[idx, term_idx] = weight
            X_transformed = transformer.transform(mtx)
            data = (usernames, X_transformed)
            logger.debug(f"{len(usernames)}, {X_transformed.shape}")
            if len(usernames) != X_transformed.shape[0]:
                raise Exception("size not match!")
            with open(f"{TOP_DIR}/var/data_svd/{key:09d}.pkl.gz", "wb") as fp:
                fp.write(gzip.compress(pickle.dumps(data)))
        except Exception as exc:
            logger.debug(exc)

    with ProcessPoolExecutor(max_workers=psutil.cpu_count()) as exe:
        for _ in tqdm(exe.map(load, args), total=len(args), desc="transforming..."):
            _
