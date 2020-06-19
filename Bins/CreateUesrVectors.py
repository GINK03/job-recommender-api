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
from typing import Set, List, Optional
from concurrent.futures import ProcessPoolExecutor
import psutil
from tqdm import tqdm
from dataclasses import dataclass
import random

HOME = Path.home()
TOP_DIR = Path(__file__).resolve().parent.parent
FILE = Path(__file__).name
try:
    sys.path.append(f"{TOP_DIR}")
    import Libs
except Exception as exc:
    raise Exception(exc)

user_dirs = []
# for target_dir in [f"{HOME}/.mnt/favs{i:02d}" for i in range(20)]:
# for target_dir in [f"{HOME}/.mnt/favs04"]:
for target_dir in [f"{HOME}/nvme0n1"]:
    for user_dir in glob.glob(f"{target_dir}/*"):
        user_dirs.append(user_dir)
random.shuffle(user_dirs) 

@dataclass
class IdxFreq:
    idx: int
    freq: float
idf = pd.read_csv(f"{TOP_DIR}/var/doc_freq.csv")
idf = idf.reset_index()
idf = {term: IdxFreq(idx, freq) for term, freq, idx in zip(idf.term, idf.freq, idf["index"])}

parser = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
def get_vector(user_dirs: List[str]) -> Libs.Tweets:
    for user_dir in tqdm(user_dirs):
        try:
            screen_name = Path(user_dir).name
            if Path(f"{TOP_DIR}/var/user_vectors/{screen_name}").exists():
                continue

            favorites_file, feeds_file = None, None
            for target_file in glob.glob(f"{user_dir}/FEEDS/*"):
                try:
                    ts_str = re.search("(\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d)", str(target_file)).group(1)
                    fetched_time = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                except ValueError as exc:
                    tb_lineno = sys.exc_info()[2].tb_lineno
                    print(f"[{FILE}] tb_lineno = {tb_lineno}, exc = {exc}", file=sys.stderr)
                    continue
                if re.search("FEEDS_\d\d\d\d", str(target_file)):
                    feeds_file = Libs.File(ts=fetched_time, typing="FEEDS", filename=target_file)
                elif re.search("FAVORITES_\d\d\d\d", str(target_file)):
                    favorites_file = Libs.File(ts=fetched_time, typing="FAVORITES", filename=target_file)
                else:
                    raise Exception("DeepError")
            if favorites_file is None or feeds_file is None:
                continue
            tweets_set = set()
            with gzip.open(favorites_file.filename, "rt") as fp:
                for line in fp:
                    try:
                        obj = json.loads(line.strip())
                        tweets_set.add(obj["text"])
                    except Exception as exc:
                        tb_lineno = sys.exc_info()[2].tb_lineno
                        print(f"[{FILE}] tb_lineno = {tb_lineno}, exc = {exc}", file=sys.stderr)
                        continue
            # もしtweets_setが300以下だったらそもそも処理しない
            if len(tweets_set) <= 300:
                continue
            with gzip.open(feeds_file.filename, "rt") as fp:
                for line in fp:
                    try:
                        obj = json.loads(line.strip())
                        tweets_set.add(obj["text"])
                    except Exception as exc:
                        tb_lineno = sys.exc_info()[2].tb_lineno
                        print(f"[{FILE}] tb_lineno = {tb_lineno}, exc = {exc}", file=sys.stderr)
                        continue

            tweets: List[str] = list(tweets_set)
            
            tweets = random.sample(tweets, min(500, len(tweets)))
            terms = []
            for tweet in tweets:
                terms += parser.parse(tweet.lower()).strip().split()
            
            tfidf = {}
            for t, f in dict(Counter(terms)).items():
                if t in idf:
                    idx = idf[t].idx
                    freq = idf[t].freq
                    tfidf[idx] = np.log1p(f)/freq
            screen_name = Path(user_dir).name
            with open(f"{TOP_DIR}/var/user_vectors/{screen_name}", "wb") as fp:
                fp.write(gzip.compress(pickle.dumps(tfidf)))
        except Exception as exc:
            tb_lineno = sys.exc_info()[2].tb_lineno
            print(f"[{FILE}] tb_lineno = {tb_lineno}, exc = {exc}", file=sys.stderr)

args = {}
for idx, user_dir in enumerate(user_dirs):
    key = idx % psutil.cpu_count()
    if key not in args:
        args[key] = []
    args[key].append(user_dir)
args = [user_dirs for user_dirs in args.values()]
# for user_dir in tqdm(user_dirs):
#    get_vector(user_dir)

with ProcessPoolExecutor(max_workers=psutil.cpu_count()) as exe:
    for _ in tqdm(exe.map(get_vector, args), total=len(args), desc="working..."):
        _

