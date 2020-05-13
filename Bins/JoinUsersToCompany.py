import glob
import json
from typing import List, Union, Set, Dict
from pathlib import Path
from dataclasses import dataclass
from Libs import Data
from Libs import OrgData
import pickle
"""
userとtweet: Setを紐付ける
Set型であるのは、
"""
username_tweets: Dict[str, Set[str]] = {}
for filename in glob.glob("var/favorites/*/*/tweets.json"):
    path = Path(filename)
    username = path.parent.parent.name
    try:
        fp = path.open()
        for line in fp:
            try:
                record = json.loads(line.strip())
                tweet = record["tweet"]
            except Exception as exc:
                print(exc)
            if username not in username_tweets:
                username_tweets[username] = set()
            username_tweets[username].add(tweet)
    except Exception as exc:
        path.unlink()
        continue




org_data: OrgData = {}
for filename in glob.glob("var/users/*"):
    username = Path(filename).name
    if username not in username_tweets:
        continue
    try:
        with open(filename) as fp:
            org = json.load(fp)["org"]
    except:
        continue
    if org not in org_data:
        org_data[org] = Data(usernames=set(), tweets=[], org=org)
    data = org_data[org]
    data.tweets.extend(list(username_tweets[username]))
    data.usernames.add(username)

with open("var/org_data.pkl", "wb") as fp:
    fp.write(pickle.dumps(org_data))
