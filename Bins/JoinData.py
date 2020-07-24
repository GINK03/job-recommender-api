import glob
import gzip
import pickle
import numpy as np
import bz2
from pathlib import Path
from tqdm import tqdm
from loguru import logger
TOP_DIR = Path(__file__).resolve().parent.parent

shards = {}
for idx, filename in enumerate(glob.glob(f"{TOP_DIR}/var/transformed/*.pkl.bz2")):
    if idx%13 not in shards:
        shards[idx%13] = []
    shards[idx%13].append(filename)

shards = [(key, filenames) for key, filenames in shards.items()]

def join_data(arg):
    key, filenames = arg
    X = None  
    users = None
    for filename in tqdm(filenames, desc="load chunks..."):
        with bz2.open(filename, 'rb') as fp:
            _users, _X = pickle.load(fp)
        if len(_users) != _X.shape[0]:
            logger.error(f"error, size not mached")
            continue
        
        if X is None:
            X = _X
            users = _users
        else:
            X = np.vstack([X, _X])
            users.extend(_users)
        logger.info(f"X.shape = {X.shape}, users_len = {len(users)}")

    np.savez_compressed(f"{TOP_DIR}/var/sources/joined_data_{key:03d}.npz", X=X, users=users)

for key, filenames in shards:
    join_data((key, filenames))
