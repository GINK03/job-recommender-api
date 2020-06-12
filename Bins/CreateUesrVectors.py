
from typing import NewType
import numpy as np
from collections import Counter
import MeCab
import json
import glob
import pandas as pd
import twint
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
for target_dir in [f"{HOME}/.mnt/favs04"]:
    for user_dir in glob.glob(f"{target_dir}/*"):
        user_dirs.append(user_dir)
random.shuffle(user_dirs) 

@dataclass
class IdxFreq:
    idx: int
    freq: float
idf = pd.read_csv(f"{HOME}/var/doc_freq.csv")
idf = idf.reset_index()
""" ヒューリスティック: 10できる """
idf = idf[idf.freq >= 10]
idf = {term: IdxFreq(idx, freq) for term, freq, idx in zip(idf.term, idf.freq, idf["index"])}

def get_vector(user_dir: str) -> Libs.Tweets:
    try:
        screen_name = Path(user_dir).name
        if Path(f"{HOME}/var/user_vectors/{screen_name}").exists():
            return None

        files = []
        for target_file in Path(user_dir).rglob("*"):
            """ .gzがファイルの末尾についていることがあり、末尾の文字列を削除する """
            name = re.sub("\.gz$", "", Path(target_file).name)
            try:
                fetched_time = datetime.datetime.strptime(name, "%Y-%m-%d %H:%M:%S")
            except ValueError as exc:
                tb_lineno = sys.exc_info()[2].tb_lineno
                print(f"[{FILE}] tb_lineno = {tb_lineno}, exc = {exc}", file=sys.stderr)
                continue
            # print(target_file, fetched_time)
            file = Libs.File(ts=fetched_time, filename=target_file.__str__())
            files.append(file)

        tweets_set: set = set([])
        for file in files:
            if re.search(r"\.gz$", file.filename):
                fp = gzip.open(file.filename, "rt")
            else:
                fp = open(file.filename, "r")

            with fp:
                for line in fp:
                    try:
                        obj = json.loads(line.strip())
                        tweets_set.add(obj["tweet"])
                    except Exception as exc:
                        tb_lineno = sys.exc_info()[2].tb_lineno
                        print(f"[{FILE}] tb_lineno = {tb_lineno}, exc = {exc}", file=sys.stderr)
                        continue
        tweets: Libs.Tweets = list(tweets_set)
        if len(tweets) < 300:
            print(f"screen_name = {screen_name}'s sample tweets is too small, skip")
            return


        parser = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
        terms = []
        for tweet in tweets:
            terms += parser.parse(tweet.lower()).strip().split()
        """
        1tweetあたりにノーマライズ
        """
        tfidf = {}
        for t, f in dict(Counter(terms)).items():
            if t in idf:
                idx = idf[t].idx
                freq = idf[t].freq
                tfidf[idx] = np.log1p(f)/freq/len(tweets)
        screen_name = Path(user_dir).name
        with open(f"{HOME}/var/user_vectors/{screen_name}", "wb") as fp:
            fp.write(gzip.compress(pickle.dumps(tfidf)))
    except Exception as exc:
        tb_lineno = sys.exc_info()[2].tb_lineno
        print(f"[{FILE}] tb_lineno = {tb_lineno}, exc = {exc}", file=sys.stderr)
# for user_dir in tqdm(user_dirs):
#    get_vector(user_dir)
with ProcessPoolExecutor(max_workers=psutil.cpu_count() * 3) as exe:
    for _ in tqdm(exe.map(get_vector, user_dirs), total=len(user_dirs)):
        _

