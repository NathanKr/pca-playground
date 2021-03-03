from sklearn.datasets import load_boston
import numpy as np
from utils import normalize_features
import matplotlib.pyplot as plt


ds_boston = load_boston()
X = ds_boston["data"]
m = X.shape[0] # number of datset points
n = X.shape[1] # number of features

X_normalized = normalize_features(X) 
sigma = (1/m)*np.dot(X_normalized.T,X_normalized) # nxn (n is 13 : number of features)
u, s, vh = np.linalg.svd(sigma)
print(f"s\n{s}")
variance_retained_vec = []
index_vec=[]
sum_s = s.sum()
for i in range(n):
    variance_retained = s[:i+1].sum()/ sum_s
    variance_retained_vec.append(100*variance_retained)
    index_vec.append(i+1)


plt.plot(index_vec,variance_retained_vec)
plt.grid()
plt.ylim(0,100)
plt.xlim(1,n)
plt.ylabel('% of variance retained')
plt.xlabel('number of pca components used')
plt.show()