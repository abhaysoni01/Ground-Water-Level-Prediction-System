import numpy as np

def adaptive_threshold(y_true, y_pred, base_quantile=0.15):
    """
    Learns a dynamic critical groundwater threshold
    instead of using a fixed expert-defined value.
    """
    base_threshold = np.quantile(y_true, base_quantile)
    error_adjustment = np.mean(abs(y_true - y_pred)) * 0.1
    return base_threshold + error_adjustment
