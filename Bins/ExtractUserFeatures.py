"""
このプログラムはUserの単語の特徴extractするプログラムで、
可視化等を目的とする
"""

import numpy as np
import pandas as pd
import glob
import bz2
import pickle
from concurrent.futures import ProcessPoolExecutor
import re
import regex
from pathlib import Path
from loguru import logger
import time


TOP_DIR = Path(__file__).resolve().parent.parent

doc_freq = pd.read_csv("../var/doc_freq.csv").reset_index()

idx_term = {idx: term for idx, term in zip(doc_freq["index"], doc_freq["term"])}


def weight_tune(w):
    for i in range(5):
        w = np.log1p(w)
    return w


def proc(filename):
    try:
        with bz2.open(filename, "rb") as fp:
            v = pickle.load(fp)
        SAMPLE_SIZE = v["__SAMPLE_SIZE__"]

        if SAMPLE_SIZE <= 100:
            return None
        del v["__SAMPLE_SIZE__"]

        a = pd.DataFrame({"idx": list(v.keys()), "w": list(v.values())})
        a["term"] = a["idx"].apply(lambda x: idx_term.get(x))
        a.sort_values(by=["w"], inplace=True, ascending=False)
        a = a[pd.notnull(a["term"])]

        a = a.head(200)
        username = Path(filename).name
        a["username"] = username
        a["w"] = a["w"].apply(weight_tune)
        # print(SAMPLE_SIZE)
        # print(a)
        return a
    except Exception as exc:
        return None


Path(TOP_DIR / "var/weights").mkdir(exist_ok=True)


def proc_wrapper(arg):
    idx, c = arg
    start = time.time()
    logger.info(f"start to {idx}")
    a = pd.concat([proc(x) for x in c if x is not None])
    a.to_csv(TOP_DIR / f"var/weights/{idx:06d}.csv.gz", index=None, compression="gzip")
    logger.info(f"end to {idx}, elapsed = {time.time() - start:0.06f}")


filenames = sorted(glob.glob(str(TOP_DIR / "var/user_vectors/*")))
a = np.array(filenames)
a = a[: len(a) // 1000 * 1000].reshape(len(a) // 1000, 1000)

with ProcessPoolExecutor(max_workers=16) as exe:
    for _ in exe.map(proc_wrapper, [(idx, c) for idx, c in enumerate(a)]):
        _
