from os.path import join 
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from utils import normalize_features

file = "ex7data1.mat"
current_dir = os.path.abspath(".")
data_dir = join(current_dir, 'data')
file_name = join(data_dir,file)
mat_dict = sio.loadmat(file_name)
print("mat_dict.keys() : ",mat_dict.keys())
X = mat_dict["X"]
m = X.shape[0]

# remove mean and feature scaling is recommended by Andrew Ng
X_normalized = normalize_features(X) 

# m = x1.size
sigma = (1/m)*np.dot(X_normalized.T,X_normalized) # nxn (n is 2 : number of features)

def plot(_X,title):
    x1 = _X[:,0]
    x2 = _X[:,1]
    plt.plot(x1,x2,'o')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid()
    plt.show()

def svd():
    u, s, vh = np.linalg.svd(sigma)
    print(f"u : {u}")
    print(f"s : {s}")
    print(f"vh : {vh}")
    variance_retained_1 = 100*s[0]/s.sum()
    print(f"% variance retained by using 1 approximated feature : {variance_retained_1}")


# def learn():
    # pca = PCA(n_components=1)
    # pca.fit(X)
    # print(pca)
    # Z = pca.transform(X)
    # print(Z.shape)
    # plt.plot(Z,'o')
    # plt.show()

plot(X,'dataset before pre processing')   
plot(X_normalized,'dataset after pre processing')   
svd()
# learn()
