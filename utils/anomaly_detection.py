
import numpy as np

"""
Docstring for utils.anomaly_detection
1. Extract features from encoder, based onthe loader 
    1.1 L2-Normalization ?    feats =feats/ np.maximum(np.linalg.norm(feats, axis = 1, keepdims=True),1e-12)
2. Extract features from normal data and calculate treshold 
3. Predict which features are anomalies 
4. Predict Batches 
"""

def feature_is_anomaly(new_sample, normal_featues, threshold):
    """
    new_sample: (m,)
    normal_features: (J,m) J is number of normal samples, m is latent dimention
    threshold
    """
    # Resturns 0 or 1 based on the threshold value
    d = np.linalg.norm(normal_featues-new_sample[None,:], axis=1).min()
    return 0 if d <= threshold else 1