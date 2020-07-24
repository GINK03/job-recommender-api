""" No. 1 """
import psutil
from concurrent.futures import ProcessPoolExecutor
import random
from tqdm import tqdm
import gzip
import MeCab
from pathlib import Path
from typing import Set, Dict
from os import environ as E
import pandas as pd
import numpy as np
import glob
import json
import re
import regex
from loguru import logger
"""
1. tweetを100万集める
2. 重複を弾くためにSet型で集計する
3. ~/.mnt/favs03/*, ~/nvme0n1 からスキャン
"""

parser = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
""" テスト用の分かち書き """
assert parser.parse("COVID19").strip().split() == ["COVID19"], "辞書ファイルが古いです"

HOME = Path.home()
TOP_DIR = Path(__file__).resolve().parent.parent

def get_tweets_from_user_dir(sub_dir):
    tweets = set()
    for filename in glob.glob(f"{sub_dir}/FEEDS/*"):
        if not Path(filename).is_file():
            logger.info(filename)
            continue
        try:
            with gzip.open(filename, "rt") as fp:
                for line in fp:
                    line = line.strip()
                    try:
                        obj = json.loads(line)
                    except:
                        continue
                    if not isinstance(obj, Dict) or obj.get("text") is None:
                        continue
                    try:
                        text = obj["text"]
                        snow_flake = re.search("/(\d{1,})$", obj["status_url"]).group(1)
                        tweets.add(text)
                    except Exception as exc:
                        logger.error(f"{exc}, {obj}")
        except Exception as exc:
            logger.error(f"{exc}")
            continue
    return tweets


user_dirs = [sub_dir for sub_dir in tqdm(glob.glob(Path("~/nvme0n1/*").expanduser().__str__()), desc="collect tweets for idf-dic")]
if E.get("TEST"):
    user_dirs = user_dirs[:100]

with ProcessPoolExecutor(max_workers=psutil.cpu_count()) as exe:
    tweets = set()
    for _tweets in tqdm(exe.map(get_tweets_from_user_dir, user_dirs), total=len(user_dirs), desc="working..."):
        tweets |= _tweets
        if len(tweets) >= 100000000:
            break

a = np.array(list(tweets), dtype="U140")
a = a[:len(a)//1000*1000].reshape((len(a)//1000, 1000))
""" IDF(doc_freq)を作成 """

def wakati(x):
    doc_freq: Dict[str, int] = {}
    for tweet in x:
        for term in set(parser.parse(tweet.lower()).strip().split()):
            if term not in doc_freq:
                doc_freq[term] = 0
            doc_freq[term] += 1
    return doc_freq

doc_freq: Dict[str, int] = {}
with ProcessPoolExecutor(max_workers=16) as exe:
    for _doc_freq in tqdm(exe.map(wakati, a), total=len(a), desc="wakati..."):
        for d, f in _doc_freq.items():
            if d not in doc_freq:
                doc_freq[d] = 0
            doc_freq[d] += f


def is_legal(x):
    # ok, e.g. akb48
    if re.search("^[a-z]{1,}[0-9]{1,}$", x):
        return x

    # filter english
    if re.search("^[a-z0-9!?]{1,}$", x):
        return None

    # check japanese
    if regex.search("^[\p{Hiragana}\p{Katakana}\p{Han}]{1,}$", x):
        return x

    return None

df = pd.DataFrame({"term": list(doc_freq.keys()), "freq": list(doc_freq.values())})
df["term"] = df["term"].astype(str).apply(is_legal)
df = df[pd.notnull(df["term"])]

df.sort_values(by=["freq"], ascending=False, inplace=True)
""" 95%上位以上を採用 """
min_freq = df.iloc[int(len(df)*0.95)].freq
df = df[df.freq > min_freq]
df.to_csv(f"{TOP_DIR}/var/doc_freq.csv", index=None)
