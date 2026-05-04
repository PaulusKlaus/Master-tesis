
import numpy as np
import torch
from scipy.spatial.distance import pdist,squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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



def extract_features_from_encoder(device, encoder, loader, l2_normalize=False):
    """
    Returns:
      feats: (N, D) numpy float32
      labels: (N,) numpy
    """
    encoder.eval()
    feats_list = []
    labels_list = []

    with torch.no_grad():   
        for batch in loader:
            if len(batch) == 3:
                x1, x2, y = batch
                x1 = x1.to(device)
                x2 = x2.to(device)
                f, z2, p1, p2 = encoder(x1, x2)
            elif len(batch) == 2:
                x1, y = batch
                x1 = x1.to(device)
                f = encoder(x1)
            else:
                raise ValueError(f"Expected batch of length 2 or 3, got {len(batch)}")

            feats_list.append(f.detach().cpu())
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
    dmins = np.linalg.norm(normal_features[None, :, :] - X[:, None, :], axis=2).min(axis=1)
    return (dmins > threshold).astype(np.int64)





def run_anomaly_detection(
            device,
            encoder,
            train_loader,
            test_loader,
            normal_class=0,
            std_factor=2.0,
            k_neighbors = 2,
            metric="euclidean",
            n_jobs=-1,
            verbose=True,
):
    """
    Fits an anomaly threshold using ONLY normal samples from train_loader,
    then predicts anomalies for samples from test_loader.
    """
    
    # 1) Extract train features 
    train_feats, train_labels = extract_features_from_encoder(device, encoder, train_loader)
    # 2) Keep only normal Features
    normal_feats = train_feats[train_labels == normal_class]

    # 3) Fit threshold via 1-NN distances within normal set ----

    nn = NearestNeighbors(n_neighbors=k_neighbors, metric=metric, n_jobs=n_jobs).fit(normal_feats)
    dists, _ = nn.kneighbors(normal_feats, return_distance=True)
    nn1 = dists[:, 1]  # exclude self-distance
    threshold = float(nn1.mean() + std_factor * nn1.std())
    if verbose:
        print("Thresholds:", threshold) #same 

    # 3) Extract test features
    test_feats, test_labels = extract_features_from_encoder(device, encoder, test_loader)
    
    test_feats_0 = test_feats[test_labels == normal_class]
    test_feats_other = test_feats[test_labels != normal_class]
    pred_0 = predict_anomaly_labels(test_feats_0, normal_feats, threshold)
    pred = predict_anomaly_labels(test_feats_other, normal_feats, threshold)
    print("Should be all 0:", pred_0)
    print("Should be all 1:", pred)

    # 5) Predict for ALL test samples
    dists_test, _ = nn.kneighbors(test_feats, n_neighbors=k_neighbors, return_distance=True)
    dmin = dists_test[:, 0]
    test_pred = (dmin > threshold).astype(np.int64)  # 0 normal, 1 anomaly
    return test_pred, test_labels, threshold
    # pred: 0 normal, 1 anomaly



def anomaly_metrics_from_multiclass(y_true, y_pred_bin, normal_class=0):
    """
    y_true: (N,) values 0..n_classes-1
    y_pred_bin: (N,) values 0/1 (0=normal, 1=anomaly)
    """
    y_true = np.asarray(y_true)
    y_pred_bin = np.asarray(y_pred_bin)

    y_true_bin = (y_true != normal_class).astype(np.int64)

    acc = accuracy_score(y_true_bin, y_pred_bin)
    cm = confusion_matrix(y_true_bin, y_pred_bin)   # [[TN, FP],[FN, TP]]
    report = classification_report(y_true_bin, y_pred_bin, target_names=["normal", "anomaly"])

    return acc, cm, report

def per_fault_detection_rate(y_true, y_pred_bin, normal_class=0):
    y_true = np.asarray(y_true)
    y_pred_bin = np.asarray(y_pred_bin)

    rates = {}
    for c in np.unique(y_true):
        if c == normal_class:
            continue
        mask = (y_true == c)
        # among true class c (a fault), how many predicted anomaly=1?
        rates[int(c)] = float((y_pred_bin[mask] == 1).mean())
    return rates