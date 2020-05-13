import glob
import json
"""
1. tweetを50万件集めてtfidfベースとする
2. 重複を弾くためにSet型で集計する
3. ~/.mnt/favs03/*からスキャン
"""
from os import environ as E
from typing import Set, Dict
from pathlib import Path
import MeCab

parser = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
""" テスト用の分かち書き """
assert parser.parse("COVID19").strip().split() == ["COVID19"], "辞書ファイルが古いです"

HOME = E.get("HOME")

tweets = set()
for sub_dir in glob.glob(f"{HOME}/.mnt/favs03/*"):
    for filename in glob.glob(f"{sub_dir}/*"):
        # print(Path(filename).is_file(), open(filename).read())
        if not Path(filename).is_file():
            print(filename)
            continue
        with open(filename) as fp:
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

    if len(tweets) >= 10_000_000:
        break

doc_freq = {}
for tweet in tweets:
    for term in set(parser.parse(tweet.lower()).strip().split()):
        if term not in doc_freq:
            doc_freq[term] = 0
        doc_freq[term] += 1

import pandas as pd

df = pd.DataFrame({"term":list(doc_freq.keys()), "freq":list(doc_freq.values())})
df.sort_values(by=["freq"], ascending=False, inplace=True)
df.to_csv("var/doc_freq.csv", index=None)
