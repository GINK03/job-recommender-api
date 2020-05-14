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
import sys
import re
from os import environ as E
from dataclasses import dataclass
import gzip
from typing import Set, List, Optional
HOME = E.get("HOME")
TOP_DIR = Path(__file__).resolve().parent.parent
try:
    sys.path.append(f"{TOP_DIR}")
    import Libs
except Exception as exc:
    raise Exception(exc)


def fetch(user: str) -> None:
    try:
        try:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            Path(f"{TOP_DIR}/tmp/{user}/{now}").mkdir(exist_ok=True, parents=True)

            c = twint.Config()
            c.Username = user
            c.Limit = 1000
            c.Following = True
            # c.Hide_output = True
            c.Store_json = True
            c.Output = f"{TOP_DIR}/tmp/{user}/{now}"
            twint.run.Favorites(c)
            print(f"finish {user}.")
        except Exception as exc:
            log = f"exc = {exc}, user = {user}"
            print(log)
    except Exception as exc:
        print(exc)


def get_tweets(user: str) -> Tweets:
    """
    favoritesを取得するのはとても時間がかかるため、datetimeを保存しておき、一ヶ月以上前のデータであれば、再取得する
    """
    target_dirs = [f"{HOME}/.mnt/favs{i:02d}" for i in range(20)]
    target_path = None
    for target_dir in target_dirs:
        if Path(f"{target_dir}/{user}").exists():
            print(f"存在しています！, {target_dir}/{user}")
            target_path = f"{target_dir}/{user}"
    if target_path is not None:
        files = []
        for target_file in Path(target_path).rglob("*"):
            """ .gzがファイルの末尾についていることがあり、末尾の文字列を削除する """
            name = re.sub("\.gz$", "", Path(target_file).name)
            fetched_time = datetime.datetime.strptime(name, "%Y-%m-%d %H:%M:%S")
            print(target_file, fetched_time)
            file = File(ts=fetched_time, filename=target_file.__str__())
            files.append(file)
        """
        File.tsで最新の情報を取り出してタイムスタンプを判定
        """
        max_file = max(files, key=lambda x: x.ts)
        if datetime.datetime.now() - max_file.ts > datetime.timedelta(days=30):
            print("古すぎるので再取得したほうが良さそうです")
            """ TODO """
        if re.search(r"\.gz$", max_file.filename):
            fp = gzip.open(max_file.filename, "rt")
        else:
            fp = open(max_file.filename, "r")

        tweets: Tweets = []
        with fp:
            for line in fp:
                obj = json.loads(line.strip())
                tweets.append(obj["tweet"])
        return tweets


idf = pd.read_csv(f"{TOP_DIR}/var/doc_freq.csv")
""" ヒューリスティック: 10できる """
idf = idf[idf.freq >= 10]
idf = {term: freq for term, freq in zip(idf.term, idf.freq)}


def calc_distances(user: str) -> pd.DataFrame:
    tweets: Tweets = get_tweets(user)
    parser = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    """
    ユーザ側のベクトルを作る
    Set型に入れて重複を避ける
    """
    tweets: Set[str] = set(tweets)
    terms: List[str] = []
    for tweet in tweets:
        terms += parser.parse(tweet.lower()).strip().split()
    """
    1tweetあたりにノーマライズ
    """
    tfidf = {}
    for t, f in dict(Counter(terms)).items():
        if t in idf:
            tfidf[t] = np.log1p(f)/idf[t]/len(tweets)

    rank: pd.DataFrame = Libs.calc_sim(tfidf)
    return rank


if __name__ == "__main__":
    """
    テストユーザ @mizchi で確認
    """
    rank = calc_distances(user="mizchi")
    print(rank)
