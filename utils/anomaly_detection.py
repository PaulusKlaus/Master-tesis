
import numpy as np
import torch
from scipy.spatial.distance import pdist,squareform
from sklearn.neighbors import NearestNeighbors

"""
Docstring for utils.anomaly_detection
1. Extract features from encoder, based onthe loader 
    1.1 L2-Normalization ?    feats =feats/ np.maximum(np.linalg.norm(feats, axis = 1, keepdims=True),1e-12)
2. Extract features from normal data and calculate treshold 
3. Predict which features are anomalies 
4. Predict Batches 



Anomaly detection pipeline :
    1. Extract latent features (as a single N x D numpy array) from your encoder

    2 Fit: compute the threshold from normal features using the nearest-neighbor distance distribution

    3. Predict: for each new sample, compute its min distance to the normal set and compare to threshold

    4. (Optional) Batch predict

"""


def extract_features_from_encoder(device, encoder, loader, use_head="z1", l2_normalize=True):
    """
    Returns:
      feats: (N, D) numpy float32
      labels: (N,) numpy
    """
    encoder.eval()
    feats_list = []
    labels_list = []

    with torch.no_grad():
        for x1, x2, y in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)

            z1, z2, p1, p2 = encoder(x1, x2)

            if use_head == "z1":
                f = z1
            elif use_head == "z2":
                f = z2
            elif use_head == "p1":
                f = p1
            elif use_head == "p2":
                f = p2
            else:
                raise ValueError("use_head must be one of: z1, z2, p1, p2")

            feats_list.append(f.detach().cpu())  # f is of the shape (batch, latent space) 
            labels_list.append(y.detach().cpu())

    feats = torch.cat(feats_list, dim=0).numpy().astype(np.float32) 
    labels = torch.cat(labels_list, dim=0).numpy()

    if l2_normalize:
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.maximum(norms, 1e-12)

    return feats, labels



def fit_nn_threshold(normal_features, k=1, std_factor=2.0, metric="euclidean"):
    """
    normal_features: (N, D) numpy
    k: use the k-th nearest neighbor distance (k=1 matches your code)
    std_factor: 2.0 matches "mean + 2*std"
    """
    # Pairwise distances
    dists = pdist(normal_features, metric=metric) # pdist is more memory effisient then pairwie distance 
    dist_matrix = squareform(dists)            # (N, N)
    sorted_cols = np.sort(dist_matrix, axis=0) # sort each column

    # sorted_cols[0] is distance to itself (0). sorted_cols[1] is 1-NN distance.
    kth_nn_dist = sorted_cols[k]               # (N,)
    threshold = float(kth_nn_dist.mean() + std_factor * kth_nn_dist.std())
    return threshold, kth_nn_dist


def fit_nn_threshold_knn(normal_features, std_factor=2.0):
    # n_neighbors=2 because the nearest neighbor of a point is itself (distance 0)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(normal_features)
    dists, _ = nn.kneighbors(normal_features, return_distance=True)
    nn1 = dists[:, 1]  # the 1-NN distance excluding itself
    threshold = float(nn1.mean() + std_factor * nn1.std())
    return threshold, nn1



def feature_is_anomaly(new_sample, normal_featues, threshold):
    """
    new_sample: (D,) or (1,D)
    normal_features: (N, D) N is number of normal samples, D is latent dimention
    returns: 0 (normal) or 1 (abnormal)
    """
    # Resturns 0 or 1 based on the threshold value
    d = np.linalg.norm(normal_featues-new_sample[None,:], axis=1).min()
    return 0 if d <= threshold else 1

def predict_anomaly_labels(features, normal_features, threshold):
    """
    features: (M, D)
    returns: (M,) array of 0/1
    """
    X = np.asarray(features, dtype=np.float32)
    # Compute min distance from each test point to the normal set
    # (M, N, D) may be large; this is simple but can be memory-heavy.
    # For big data, chunk it (see below).
    dmins = np.linalg.norm(normal_features[None, :, :] - X[:, None, :], axis=2).min(axis=1)
    return (dmins > threshold).astype(np.int64)



