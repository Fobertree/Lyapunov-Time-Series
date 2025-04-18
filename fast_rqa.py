# Recurrence Quantification Analysis
# DO NOT do import * from here, keep 'global' scope to this module
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import ParameterGrid
import sys
import time
from tqdm import tqdm
import concurrent.futures

print(sys.version) # GIL removed 3.13, better to ThreadPool than ProcessPool
# only using ProcessPool since my python is only 3.10.3
# knn imputation from sklearn

# --- Parameters ---
epsilon = 0.3     # Recurrence threshold
tau = 1           # Time delay
d_E = 6          # Embedding dimension
W = 3             # Theiler window (minimum separation)
m = 5            # Time evolution step size
window = 13      # Window size for Lyapunov time series
min_points = d_E + 1  # Min neighbors to fit log divergence

X_embedded = None
num_vectors = None
kd = None

def init_pool(X_embedded_val, num_vectors_val, kd_obj):
    # Initialize pool processes global varibales: spaghetti but its just how this works
    global X_embedded, num_vectors, kd
    X_embedded = X_embedded_val
    num_vectors = num_vectors_val
    kd = kd_obj


def rqa_helper(start):
    # func must be top-level to work
    global X_embedded, num_vectors, kd
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
            return np.mean(dists) / (m * tau)
            # local_exponents.append(np.mean(dists) / (m * tau))

    if len(local_exponents) > 0:
        return np.mean(local_exponents)
        # lyap_timeseries.append(np.mean(local_exponents))
    
    return np.nan

def rqa(X, impute = False):
    # --- Phase space reconstruction ---
    global X_embedded, num_vectors, kd
    
    # min-max scale [0,1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.reshape(-1, 1)).flatten()

    print(X)
    
    N = len(X)

    X_embedded = np.array([X[i:i + d_E * tau:tau] for i in range(N - (d_E * tau - 1))])
    num_vectors = len(X_embedded)

    # --- Build KD-Tree once ---
    kd = KDTree(X_embedded)

    # --- Time series of local Lyapunov exponents ---
    lyap_timeseries = [np.nan for _ in range(num_vectors - m - window)]

    with concurrent.futures.ProcessPoolExecutor(initializer=init_pool, initargs= (X_embedded, num_vectors, kd,),max_workers=4) as executor:
        futures = {executor.submit(rqa_helper, start): start for start in range(0, num_vectors - m - window)}
    
    for future in tqdm(concurrent.futures.as_completed(futures)):
        try:
            res = future.result()
            i = futures[future]
            lyap_timeseries[i] = res
        except Exception as e:
            print(f"FAILED:{e}, {future}")

    # wait all    
    executor.shutdown(wait=True)
    lyap_timeseries = np.array(lyap_timeseries)
    
    if impute:
        is_imputed = np.isnan(lyap_timeseries)
        imputer = KNNImputer(missing_values=np.nan, n_neighbors=3) # n_neighbors may be hyperparameter
        lyap_timeseries = imputer.fit_transform(lyap_timeseries.reshape(-1, 1)).flatten()

        return (lyap_timeseries, is_imputed)
    
    return lyap_timeseries

def optimize(X, param_grid=None, objective_fn=None, impute=True, verbose=True):
    """
    Optimize RQA parameters using grid search.

    Parameters:
        X: np.array - 1D time series input
        param_grid: dict - dictionary of parameter ranges
        objective_fn: callable - scoring function taking lyap_series, imputed_mask → score
        impute: bool - whether to use imputation in rqa
        verbose: bool - print results or not

    Returns:
        best_params: dict - parameters that maximize the objective
        best_score: float
        best_series: np.array - best Lyapunov series found
    """
    
    # this is hella stupid but i dont have the time to find a better approach
    if param_grid is None:
        param_grid = {
            'epsilon': [0.2, 0.3, 0.4],
            'tau': [1],
            'd_E': [4, 6],
            'W': [2, 3],
            'm': [3, 5],
            'window': [10, 13],
        }
        print("Set param grid to default")

    if objective_fn is None:
        def objective_fn(lyap, mask):
            # maximize stdev, penalize imputations/nan
            return np.nanstd(lyap) - 0.5 * np.sum(mask) / len(lyap)

    best_score = -np.inf
    best_params = None
    best_imputed = None # corresponding to best_series
    best_series = None

    for params in ParameterGrid(param_grid):
        # Set module-level globals to be picked up in rqa()
        globals().update(params)
        globals()["min_points"] = params["d_E"] + 1

        try:
            result = rqa(X, impute=impute)
            if impute:
                lyap_series, imputed_mask = result
            else:
                lyap_series = result
                imputed_mask = np.isnan(lyap_series)

            score = objective_fn(lyap_series, imputed_mask)

            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_imputed = imputed_mask.copy()
                best_series = lyap_series.copy()

            if verbose:
                print(f"Params: {params} | Score: {score:.4f}")

        except Exception as e:
            if verbose:
                print(f"Params {params} failed: {e}")
            continue
        
    return best_series, best_imputed, best_params, best_score, 

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
