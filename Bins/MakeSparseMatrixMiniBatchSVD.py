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
from loguru import logger

logger.add(sys.stdout, format="{filename} {name} {time} {level} {message}", filter="my_module", level="DEBUG")

HOME = Path.home()
FILE = Path(__file__).name
TOP_DIR = Path(__file__).resolve().parent.parent

idf = pd.read_csv(f'{TOP_DIR}/var/doc_freq.csv')
idf = idf[:1000000]
WORD_SIZE = len(idf)

if "--create_transformer" in sys.argv:
    SAMPLE_SIZE = 10000
    logger.info(f"total word size is = {WORD_SIZE}")
    start_time = time.time() 
    def load(arg):
        idx, filename = arg
        try:
            with open(filename, "rb") as fp:
                vec = pickle.loads(gzip.decompress(fp.read()))
            return (idx, vec)
        except Exception as exc:
            logger.debug(f"{exc}, {idx}, {filename}")
            return None
    
    transformer = TruncatedSVD(n_components=500, n_iter=5, random_state=0)

    input_user_files = glob.glob(f"{TOP_DIR}/var/user_vectors/*")
    BATCH_SIZE = 900000
    for mini_batch in range(1):
        mtx = lil_matrix((BATCH_SIZE, WORD_SIZE))
        args = []
        for idx, filename in tqdm(enumerate(random.sample(input_user_files, BATCH_SIZE)), desc="load example users..."):
            args.append((idx, filename))
        with ProcessPoolExecutor(max_workers=psutil.cpu_count()) as exe:
            for ret in tqdm(exe.map(load, args), total=len(args), desc="load example users..."):
                if ret is None:
                    continue
                idx, vec = ret
                for term_idx, weight in vec.items():
                    if term_idx >= WORD_SIZE:
                        continue
                    mtx[idx, term_idx] = weight
        joblib.dump(mtx, f"{TOP_DIR}/var/mtx_{mini_batch:03d}.joblib", compress="gzip")

        logger.debug(f"[{FILE}] mini_batch = {mini_batch}, start to train TruncatedSVD...")
        transformer.fit(mtx)
        elapsed_time = time.time() - start_time
        logger.debug(f"[{FILE}] mini_batch = {mini_batch}, elapsed_time = {elapsed_time}")
    
    logger.debug(f"[{FILE}] start to transform matrix...")
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
