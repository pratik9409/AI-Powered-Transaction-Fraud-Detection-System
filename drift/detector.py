import numpy as np
from scipy.stats import ks_2samp
from sklearn.covariance import MinCovDet
import warnings

class ConceptDriftDetector:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.reference_window = None
        self.current_window = []
        self.drift_count = 0
    
    def add_data(self, features):
        if self.reference_window is None:
            if len(self.current_window) < self.window_size:
                self.current_window.append(features)
            else:
                self.reference_window = np.array(self.current_window)
                self.current_window = []
        else:
            if len(self.current_window) < self.window_size:
                self.current_window.append(features)
            else:
                self._test_for_drift()
                self.current_window = []
    
    def _test_for_drift(self):
        current_data = np.array(self.current_window)
        
        # 1. Kolmogorov-Smirnov test for each feature
        p_values = []
        for i in range(self.reference_window.shape[1]):
            try:
                _, p_value = ks_2samp(self.reference_window[:, i], current_data[:, i])
                p_values.append(p_value)
            except:
                p_values.append(1.0)
        
        # 2. Covariance shift detection
        robust_cov = MinCovDet().fit(self.reference_window)
        try:
            cov_score = robust_cov.mahalanobis(current_data).mean()
            cov_threshold = robust_cov.mahalanobis(self.reference_window).mean() * 1.5
        except:
            cov_score = 0
            cov_threshold = 0
        
        # Combined decision
        significant_drift = any(p < 0.01 for p in p_values) or cov_score > cov_threshold
        
        if significant_drift:
            self.drift_count += 1
            if self.drift_count >= 3:  # Persistent drift
                self._alert_drift()
                self.drift_count = 0
    
    def _alert_drift(self):
        # In practice, this would trigger model retraining
        print("Warning: Significant concept drift detected!")
        # Could integrate with AutoML retraining
        