


"""
コサイン類似度を計算する
"""

import pickle
from pathlib import Path
from typing import Dict
import pandas as pd

TOP_DIR = Path(__file__).resolve().parent.parent
with open(f"{TOP_DIR}/var/org_tfidf.pkl", "rb") as fp:
    org_tfidf = pickle.load(fp)

def calc_sim(tfidf_s: Dict[str, float]) -> pd.DataFrame:
    """
    入力されたtfidfをもとに、ターゲットとの類似度を計算して降順で返す
    Args:
        - tfidf_s: sourceとなる tfidf
    Returns:
        - ret: 類似度で降順になったpd.DataFrame
    """
    ret = []
    for org, tfidf_t in org_tfidf.items():
        same_terms = set(tfidf_s.keys()) & set(tfidf_t.keys())
        sim = 0.0
        for term in same_terms:
            sim += tfidf_s[term] * tfidf_t[term]
        ret.append( {"organization":org, "similarity":sim} ) 
    ret: pd.DataFrame = pd.DataFrame(ret) 
    ret.sort_values(by=["similarity"], inplace=True, ascending=False)
    return ret
