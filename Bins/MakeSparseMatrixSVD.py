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
import bz2
from loguru import logger

HOME = Path.home()
FILE = Path(__file__).name
TOP_DIR = Path(__file__).resolve().parent.parent

def wight_tune(w):
    for i in range(5):
        w = np.log1p(w)
    return w

idf = pd.read_csv(f'{TOP_DIR}/var/doc_freq.csv')
WORD_SIZE = 1000000

if "--create_transformer" in sys.argv:
    SAMPLE_SIZE = 1000000
    logger.info(f"total word size is = {WORD_SIZE}")
    start_time = time.time() 

    def load(arg):
        filename = arg
        try:
            with bz2.open(filename, "rb") as fp:
                vec = pickle.load(fp)

            SAMPLE_SIZE = vec["__SAMPLE_SIZE__"]
            del vec["__SAMPLE_SIZE__"]
            if SAMPLE_SIZE < 100:
                return None
            return (vec)
        except Exception as exc:
            logger.error(f"{exc}, {filename}")
            Path(filename).unlink()
            return None
    args = []
    for idx, filename in tqdm(enumerate(glob.glob(f"{TOP_DIR}/var/user_vectors/*")[:SAMPLE_SIZE]), desc="load example users..."):
        args.append(filename)
    
    mtx = lil_matrix((SAMPLE_SIZE, WORD_SIZE))

    counter = 0
    with ProcessPoolExecutor(max_workers=psutil.cpu_count()) as exe:
        for ret in tqdm(exe.map(load, args), total=len(args), desc="load example users..."):
            if ret is None:
                continue
            vec = ret
            for term_idx, weight in vec.items():
                if term_idx >= WORD_SIZE:
                    continue
                mtx[counter, term_idx] = wight_tune(weight)
            counter += 1

    logger.info(mtx.shape)
    mtx = mtx[:counter]
    logger.info(mtx.shape)
    # exit()
    logger.info(f"[{FILE}] start to train TruncatedSVD...")
    transformer = TruncatedSVD(n_components=500, n_iter=10, random_state=0)
    transformer.fit(mtx)
    elapsed_time = time.time() - start_time
    logger.info(f"[{FILE}] elapsed_time = {elapsed_time}")
    logger.info(f"[{FILE}] start to transform matrix...")
    X_transformed = transformer.transform(mtx[:5000])
    logger.info(X_transformed)
    logger.info(X_transformed.shape)
    logger.info(type(X_transformed))
    joblib.dump(transformer, f"{TOP_DIR}/var/transformer.joblib")

if "--transform" in sys.argv:
    transformer = joblib.load(f"{TOP_DIR}/var/transformer.joblib")
    """ 1000個づつ分割 """
    filenames = glob.glob(f"{TOP_DIR}/var/user_vectors/*")
    args = []
    STEP = 4000
    for i in range(0, len(filenames), STEP):
        args.append((i, filenames[i:i+STEP]))
   
    Path(f"{TOP_DIR}/var/transformed").mkdir(exist_ok=True, parents=True)
    def load(arg):
        key, filenames = arg
        mtx = lil_matrix((STEP, WORD_SIZE))
        usernames = []
        counter = 0
        for idx, filename in enumerate(filenames):
            try:
                with bz2.open(filename, "rb") as fp:
                    vec = pickle.load(fp)
            except Exception as exc:
                tb_lineno = sys.exc_info()[2].tb_lineno
                logger.error(f"[{FILE}] exc = {exc}, tb_lineno = {tb_lineno}")
                continue
            SAMPLE_SIZE = vec["__SAMPLE_SIZE__"]
            del vec["__SAMPLE_SIZE__"]
            if SAMPLE_SIZE < 100:
                continue
            
            for term_idx, weight in vec.items():
                if term_idx >= 1000000:
                    continue
                mtx[counter, term_idx] = weight
            usernames.append(Path(filename).name)
            counter += 1
        mtx = mtx[:counter]
        X_transformed = transformer.transform(mtx)
        data = (usernames, X_transformed)
        logger.info(f"{len(usernames)}, {X_transformed.shape}")
        if len(usernames) != X_transformed.shape[0]:
            raise Exception("size not match!")
        with bz2.open(f"{TOP_DIR}/var/transformed/{key:09d}.pkl.bz2", "wb") as fp:
            fp.write(pickle.dumps(data))

    with ProcessPoolExecutor(max_workers=psutil.cpu_count()) as exe:
        for _ in tqdm(exe.map(load, args), total=len(args), desc="transforming..."):
            _
