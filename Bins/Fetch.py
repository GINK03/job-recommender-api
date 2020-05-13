import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path
import json
from typing import Set, Union, List, Tuple, NewType
from concurrent.futures import ProcessPoolExecutor
import sys

FILE = Path(__file__).name

def get_orgs_from_qiita_list() -> List:
    """
    Qiitaから最新のオーガニゼーションデータを取得し、Listとして返す
    NOTE: 53は現存する最大値
    Returns:
        - orgs: オーガニゼーションの名前名を含んだList
    """
    urls = [f"https://qiita.com/organizations?page={i}" for i in range(1, 53)]
    orgs: Union[Set, List] = set()
    for url in urls:
        with requests.get(url) as r:
            html: str = r.text 
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", {"href": re.compile("/organizations/(.{1,})$")}):
            orgs.add(Path(a.get("href")).name)
    orgs: List = list(orgs)
    return orgs

orgs = get_orgs_from_qiita_list()

def get_user_record(org: str) -> None:
    """
    Inputs:
        - org: 組織名
    Outputs:
        - username.json: ユーザ名、ツイッターへのリンク、組織名を含んだjson
    """
    url = f"https://qiita.com/organizations/{org}/members"
    print(url)
    with requests.get(url) as r:
        html: str = r.text 
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", {"href": True}):
        href = a.get("href")
        if href.count('/') == 1 and "?" not in href and href not in {"/", "/about", "/terms", "/privacy", "/users", "/organizations", "/advent-calendar"}:
            try:
                user_link = f"https://qiita.com{href}"
                username = user_link.split("/")[-1]
                if Path(f'var/users/{username}').exists():
                    continue
                with requests.get(user_link, timeout=60) as r:
                     html = r.text 
                soup: BeautifulSoup = BeautifulSoup(html, "lxml")
                sns_link_list: Union[BeautifulSoup, None] = soup.find(attrs={"class":re.compile("SnsLinkList")})
                if sns_link_list and sns_link_list.find("a", {"href":re.compile("https://twitter.com")}):
                    twitter_link = sns_link_list.find("a", {"href":re.compile("https://twitter.com")}).get("href")
                    record = {"org":org, "user_link": user_link, "twitter_link": twitter_link}
                    with open(f'var/users/{username}', 'w') as fp:
                        fp.write(json.dumps(record, indent=2))
            except Exception as exc:
                tb_lineno = sys.exc_info()[2].tb_lineno
                print(f"[{FILE}] exc = {exc}, tb_lineno = {tb_lineno}", file=sys.stderr)

# [get_user_record(org) for org in orgs]
with ProcessPoolExecutor(max_workers=24) as exe:
    exe.map(get_user_record, orgs)

