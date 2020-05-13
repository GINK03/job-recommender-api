import json
from typing import Union, Dict, Set, List, NewType
from dataclasses import dataclass

@dataclass
class Data:
    usernames: Set[str]
    tweets: List[str]
    org: str

OrgData = NewType("OrgData", Dict[str, Data])
