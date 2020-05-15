import numpy as np

import faiss

npz = np.load("../tmp/joined_data.npz")
users = npz["users"]
X = npz["X"].astype(np.float32) 
X = np.nan_to_num(X)
#X[:, 0] += np.arange(X.shape[0]) / 1000.
"""
index = faiss.IndexFlatL2(100)
print(index.is_trained)
print(X.shape)
index.add(X)
print(index.ntotal)

uq = users[:10]
xq = X[:10]
#print(xq)
print(uq)
index.nprobe = 10
D, I = index.search(xq, k=4)
print(users[I])
"""
X = X/np.linalg.norm(X, axis=-1)[:, np.newaxis]
from numpy.linalg import norm
xq = X[:1]
uq = users[:1]
r = np.dot(X,xq.T)
r = np.nan_to_num(r)
idx = r.flatten().argsort()[-10:][::-1]
print(r)
print(idx)
print(r[idx])
print(uq)
print(users[idx])
