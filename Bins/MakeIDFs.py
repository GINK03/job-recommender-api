import pandas as pd
import glob
import json

"""
1. tweetを100万集める
2. 重複を弾くためにSet型で集計する
3. ~/.mnt/favs03/*からスキャン
"""
from os import environ as E
from typing import Set, Dict
from pathlib import Path
import MeCab
import gzip
from tqdm import tqdm

parser = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
""" テスト用の分かち書き """
assert parser.parse("COVID19").strip().split() == ["COVID19"], "辞書ファイルが古いです"

HOME = Path.home()

tweets = set()
for sub_dir in tqdm(glob.glob(f"{HOME}/.mnt/favs03/*"), desc="collect tweets for idf-dic"):
    for filename in glob.glob(f"{sub_dir}/FEED/*"):
        print(Path(filename).is_file(), open(filename).read())
        if not Path(filename).is_file():
            print(filename)
            continue
        with gzip.open(filename, "rt") as fp:
            for line in fp:
                line = line.strip()
                try:
                    obj = json.loads(line)
                except:
                    continue
                if not isinstance(obj, Dict) or obj.get("tweet") is None:
                    continue
                try:
                    tweet = obj["tweet"]
                    tweets.add(tweet)
                except Exception as exc:
                    print(obj)
    if len(tweets) >= 10000000:
        break
doc_freq = {}
for tweet in tweets:
    for term in set(parser.parse(tweet.lower()).strip().split()):
        if term not in doc_freq:
            doc_freq[term] = 0
        doc_freq[term] += 1


df = pd.DataFrame({"term": list(doc_freq.keys()), "freq": list(doc_freq.values())})
df.sort_values(by=["freq"], ascending=False, inplace=True)
df.to_csv("var/doc_freq.csv", index=None)
