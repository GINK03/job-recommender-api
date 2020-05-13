import glob
import json
import os
from pathlib import Path
from subprocess import Popen
from subprocess import PIPE
from tqdm import tqdm
import random
import pandas as pd
from threading import stack_size
import datetime
import resource
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import twint

FILE = Path(__file__).name


def process(user: str) -> None:
    try:
        try:
            if Path(f"var/favorites/{user}/err.log").exists():
                return None

            #print(f"[{FILE}] start to {user}...")
            Path(f"var/favorites/{user}").mkdir(exist_ok=True, parents=True)
            dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            c = twint.Config()
            c.Username = user
            c.Limit = 1000
            # c.User_full = True
            c.Following = True
            c.Hide_output = True
            c.Store_json = True
            c.Output = f"var/favorites/{user}/{dt}"
            twint.run.Favorites(c)
            print(f"[{FILE}] finish {user}.")
        except Exception as exc:
            log = f"[{FILE}] exc = {exc}, user = {user}"
            with open(f"var/favorites/{user}/err.log", 'w') as fp:
                fp.write(log)
    except Exception as exc:
        print(exc)
if __name__ == "__main__":
    """
    このファイル単体で実行した時、すべてのユーザのデータを取得する
    """
    screen_names = []
    for filename in glob.glob('var/users/*'):
        try:
            obj = json.load(open(filename))
        except Exception as exc:
            print(exc)
            continue
        user_link = obj["user_link"]
        screen_name = user_link.split("/")[-1]
        screen_names.append(screen_name)
    
    while True:
        random.shuffle(screen_names)
        with ProcessPoolExecutor(max_workers=100) as exe:
            for r in tqdm(exe.map(process, screen_names), total=len(screen_names)):
                r
