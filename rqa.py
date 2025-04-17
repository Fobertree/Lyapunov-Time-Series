# Recurrence Quantification Analysis

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import sys
import time

print(sys.version)
# knn imputation from sklearn


# --- Parameters ---
epsilon = 0.5     # Recurrence threshold
tau = 1           # Time delay
d_E = 10          # Embedding dimension
W = 20            # Theiler window (minimum separation)
m = 20            # Time evolution step size
window = 100      # Window size for Lyapunov time series
min_points = d_E + 1  # Min neighbors to fit log divergence

def rqa(X, impute = False):
    # --- Phase space reconstruction ---

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.reshape(-1, 1)).flatten()

    X_embedded = np.array([X[i:i + d_E * tau:tau] for i in range(N - (d_E * tau - 1))])
    num_vectors = len(X_embedded)

    # --- Build KD-Tree once ---
    kd = KDTree(X_embedded)

    # --- Time series of local Lyapunov exponents ---
    lyap_timeseries = []

    for start in range(0, num_vectors - m - window):
        local_exponents = []
        
        # Analyze each point in the current window
        for i in range(start, start + window):
            # Find recurrence neighbors within epsilon
            neighbors_i = kd.query_radius([X_embedded[i]], r=epsilon)[0]
            
            # Filter valid neighbors based on W and m
            valid_neighbors = [
                j for j in neighbors_i 
                if abs(j - i) > W and (j + m) < num_vectors and (i + m) < num_vectors
            ]
            
            if len(valid_neighbors) < min_points:
                continue  # Skip if not enough neighbors

            # Measure divergence over time
            dists = []
            for j in valid_neighbors:
                dist_0 = np.linalg.norm(X_embedded[j] - X_embedded[i])
                dist_m = np.linalg.norm(X_embedded[j + m] - X_embedded[i + m])
                if dist_0 > 0 and dist_m > 0:
                    dists.append(np.log(dist_m / dist_0))
            
            if len(dists) >= min_points:
                local_exponents.append(np.mean(dists) / (m * tau))

        if len(local_exponents) > 0:
            lyap_timeseries.append(np.mean(local_exponents))
        else:
            lyap_timeseries.append(np.nan)  # impute via interpolation later
    
    if impute:
        is_imputed = np.isnan(lyap_timeseries)
        lyap_timeseries = KNNImputer(missing_values=np.nan, n_neighbors=3) # n_neighbors may be hyperparameter?

        return (lyap_timeseries, is_imputed)
    
    return lyap_timeseries

if __name__ == "__main__":
    np.random.seed(42)
    N = 1000
    X = np.random.randn(N)
    start_time = time.time()
    lyap_timeseries = rqa(X)
    print(f"{time.time() - start_time} seconds")

    # --- Plotting ---
    plt.figure(figsize=(10, 4))
    plt.plot(lyap_timeseries, label="Local MLE (RQA-based)", color='teal')
    plt.xlabel("Time Index")
    plt.ylabel("Lyapunov Exponent")
    plt.title("Local Lyapunov Exponent Time Series via RQA")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
