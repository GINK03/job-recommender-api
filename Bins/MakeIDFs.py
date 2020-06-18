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
import glob
import json

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
            print(filename)
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
                        tweets.add(text)
                    except Exception as exc:
                        print(exc, obj)
        except Exception as exc:
            print(exc)
            continue
    doc_freq = {}
    for tweet in tweets:
        for term in set(parser.parse(tweet.lower()).strip().split()):
            if term not in doc_freq:
                doc_freq[term] = 0
            doc_freq[term] += 1
    return doc_freq


""" IDF(doc_freq)を作成 """
doc_freq: Dict[str, int] = {}
user_dirs = [sub_dir for sub_dir in tqdm(glob.glob(f"{HOME}/nvme0n1/*"), desc="collect tweets for idf-dic")]
if E.get("TEST"):
    user_dirs = user_dirs[:1000000]

with ProcessPoolExecutor(max_workers=psutil.cpu_count()) as exe:
    for sub_doc_freq in tqdm(exe.map(get_tweets_from_user_dir, user_dirs), total=len(user_dirs), desc="working..."):
        for term, freq in sub_doc_freq.items():
            if term not in doc_freq:
                doc_freq[term] = 0
            doc_freq[term] += freq


df = pd.DataFrame({"term": list(doc_freq.keys()), "freq": list(doc_freq.values())})
df.sort_values(by=["freq"], ascending=False, inplace=True)
""" 95%上位以上を採用 """
min_freq = df.iloc[int(len(df)*0.95)].freq
df = df[df.freq > min_freq]
df.to_csv(f"{TOP_DIR}/var/doc_freq.csv", index=None)
