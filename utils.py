from sklearn.preprocessing import StandardScaler

def normalize_features(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X_normalized = scaler.transform(X) # mean is removed and feature are scaled
    return X_normalized