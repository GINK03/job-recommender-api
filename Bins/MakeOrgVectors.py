import pickle
from Libs import OrgData
import MeCab
from collections import Counter
import pandas as pd
import numpy as np
from datetime import datetime
parser = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
""" テスト用の分かち書き """
assert parser.parse("COVID19").strip().split() == ["COVID19"], "辞書ファイルが古いです"

with open('var/org_data.pkl', 'rb') as fp:
    org_data: OrgData = pickle.load(fp)

idf = pd.read_csv("var/doc_freq.csv")
""" ヒューリスティック: 10できる """
idf = idf[idf.freq >= 10]
idf = {term:freq for term, freq in zip(idf.term, idf.freq)}

org_tfidf = {}
for org, data in org_data.items():
    terms = []
    for tweet in data.tweets:
        terms += parser.parse(tweet).strip().split()
    tf = dict(Counter(terms))
  
    """
    1tweetあたりのウェイトにノーマライズする
    """
    tfidf = {}
    for t in list(tf.keys()):
        if t in idf:
            tfidf[t] = np.log1p(tf[t])/idf[t] / len(data.tweets)
    org_tfidf[org] = tfidf

with open('var/org_tfidf.pkl', 'wb') as fp:
    fp.write(pickle.dumps(org_tfidf))
