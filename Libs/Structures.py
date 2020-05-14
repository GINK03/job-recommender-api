import json
from typing import Union, Dict, Set, List, NewType
from dataclasses import dataclass
import datetime
@dataclass
class Data:
    usernames: Set[str]
    tweets: List[str]
    org: str

OrgData = NewType("OrgData", Dict[str, Data])

@dataclass
class File:
    ts: datetime.datetime
    filename: str


Tweets = NewType("Tweets", List[str])
