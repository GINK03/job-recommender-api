""" STEP 2 """
from typing import NewType
import numpy as np
from collections import Counter
import MeCab
import json
import glob
import pandas as pd
from pathlib import Path
import datetime
import pickle
import sys 
import re 
from os import environ as E
from dataclasses import dataclass
import gzip
import zlib
from typing import Set, List, Optional
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import psutil
from tqdm import tqdm
from dataclasses import dataclass
import random
import socket
from loguru import logger
import bz2
import time

logger.add(sys.stdout, format="{file} {time} {level} {message} {line}", filter="placeholder", level="DEBUG")

HOME = Path.home()
TOP_DIR = Path(__file__).resolve().parent.parent
FILE = Path(__file__).name

try: 
    sys.path.append(f"{TOP_DIR}")
    import Libs
except Exception as exc:
    raise Exception(exc)



@dataclass
class IdxFreq:
    idx: int
    freq: float


idf = pd.read_csv(f"{TOP_DIR}/var/doc_freq.csv")
idf = idf.reset_index()
idf = {term: IdxFreq(idx, freq) for term, freq, idx in zip(idf.term, idf.freq, idf["index"])}

parser = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

def get_vector(user_dirs: List[str]) -> Libs.Tweets:
    for user_dir in tqdm(user_dirs, disable=True):
        try:
            screen_name = Path(user_dir).name
            if Path(f"{TOP_DIR}/var/user_vectors/{screen_name}").exists():
                continue

            tweets_set = set()
            feeds = glob.glob(f"{user_dir}/FEEDS/*.gz")
            for feed in feeds:
                with gzip.open(feed, "rt") as fp:
                    try:
                        for line in fp:
                            try:
                                obj = json.loads(line.strip())
                                if screen_name in obj["status_url"].lower():
                                    tweets_set.add(obj["text"])
                            except Exception as exc:
                                tb_lineno = sys.exc_info()[2].tb_lineno 
                                logger.debug(f"[{FILE}] tb_lineno = {tb_lineno}, exc_type = {type(exc)}, exc = {exc}")
                                continue
                    except gzip.BadGzipFile:
                        Path(feed).unlink()
                    except zlib.error:
                        Path(feed).unlink()

            tweets: List[str] = list(tweets_set)
            tweets = random.sample(tweets, min(500, len(tweets)))
            # print(tweets)
            terms = []
            for tweet in tweets:
                terms += parser.parse(tweet.lower()).strip().split()

            tfidf = {"__SAMPLE_SIZE__": len(tweets)}
            for t, f in dict(Counter(terms)).items():
                if t in idf:
                    idx = idf[t].idx
                    freq = idf[t].freq
                    tfidf[idx] = np.log1p(f) / freq
            with bz2.open(f"{TOP_DIR}/var/user_vectors/{screen_name}", "wb") as fp:
                fp.write(pickle.dumps(tfidf))
            logger.info(f"finish sample and compress of {screen_name}")
        except Exception as exc:
            tb_lineno = sys.exc_info()[2].tb_lineno
            logger.debug(f"[{FILE}] tb_lineno = {tb_lineno}, exc_type = {type(exc)}, exc = {exc}")

def load_files(target_dir):
    start = time.time()
    logger.info(f"start to {target_dir}...")
    files =  glob.glob(f"{target_dir}/*")
    logger.info(f"end to {target_dir}, elapsed = {time.time() - start:0.06f}")
    return files

target_dirs = [f"{HOME}/.mnt/nfs/favs{i:02d}" for i in range(0,8)]
with ThreadPoolExecutor(max_workers=16) as exe:
    user_dirs = [a for a in exe.map(load_files, target_dirs)]    
arg = np.hstack(np.array(user_dirs))
print(arg.shape)
np.random.shuffle(arg)
cpu_count = psutil.cpu_count()
arg = arg[:len(arg)//1000//cpu_count*1000*cpu_count].reshape((len(arg)//1000//cpu_count, cpu_count, 1000))
# for user_dirs in arg:
#    get_vector(user_dirs)
for batch in arg: 
    with ProcessPoolExecutor(max_workers=psutil.cpu_count()) as exe:
        for _ in tqdm(exe.map(get_vector, batch), total=len(batch), desc="working..."):
            _


