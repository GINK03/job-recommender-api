import pandas as pd
import twint
from pathlib import Path
import datetime
def process(user: str) -> None:
    try:
        try:
            Path(f"tmp/{user}").mkdir(exist_ok=True, parents=True)
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            c = twint.Config()
            c.Username = user
            c.Limit = 1000
            c.Following = True
            c.Hide_output = True
            c.Store_json = True
            c.Output = f"tmp/{user}/{now}"
            twint.run.Favorites(c)
            print(f"finish {user}.")
        except Exception as exc:
            log = f"exc = {exc}, user = {user}"
            print(log)
    except Exception as exc:
        print(exc)


user = "TJO_datasci"
process(user)

idf = pd.read_csv("var/doc_freq.csv")
""" ヒューリスティック: 10できる """
idf = idf[idf.freq >= 10]
idf = {term:freq for term, freq in zip(idf.term, idf.freq)}

"""
ユーザ側のベクトルを作る
"""
import glob

for filename in glob.glob("tmp/TJO_datasci/*/*"):
    


