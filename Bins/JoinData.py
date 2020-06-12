import glob
import gzip
import pickle
import numpy as np

from pathlib import Path

TOP_DIR = Path(__file__).resolve().parent.parent
X = None  
users = None
for filename in glob.glob(f"{TOP_DIR}/tmp/data_svd/*.pkl.gz"):
    with open(filename, 'rb') as fp:
        _users, _X = pickle.loads(gzip.decompress((fp.read())))
        if len(_users) != _X.shape[0]:
            continue
        # print(users)
        if X is None:
            X = _X
            users = _users
        else:
            X = np.vstack([X, _X])
            users.extend(_users)
        print(X.shape, len(users))

np.savez(f"{TOP_DIR}/tmp/joined_data_svd.npz", X=X, users=users)
