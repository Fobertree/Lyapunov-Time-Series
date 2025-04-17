import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

# === Parameters ===
N = 1000
X = np.random.randn(N)              # Raw time series
epsilon = 3                         # Neighborhood radius
tau = 1                             # Time delay
d_E = 10                            # Embedding dimension
W = 20                              # Minimum time separation (Theiler window)
m = 20                              # Evolution steps

# min-max [0,1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, 1)).flatten()

# === Time delay embedding ===
X_embedded = np.array([
    X[i:i + d_E * tau:tau] for i in range(N - (d_E * tau - 1))
])

# === Build KDTree for neighbor lookup ===
kd = KDTree(X_embedded)
neighbors = kd.query_radius(X_embedded, r=epsilon)

# === Lyapunov spectrum time series ===
lyap_timeseries = []

for i in range(0, len(X_embedded) - m):
    neighbors_i = neighbors[i]
    y_list, z_list = [], []

    for j in neighbors_i:
        # Filter neighbors by Theiler window and bounds
        if abs(j - i) > W and (j + m) < len(X_embedded) and (i + m) < len(X_embedded):
            y_j = X_embedded[j] - X_embedded[i]
            z_j = X_embedded[j + m] - X_embedded[i + m]
            y_list.append(y_j)
            z_list.append(z_j)

    # Only compute A_i if enough neighbors are available
    if len(y_list) >= d_E:
        Y = np.stack(y_list, axis=1)
        Z = np.stack(z_list, axis=1)
        V = Y @ Y.T / Y.shape[1]
        C = Z @ Y.T / Y.shape[1]
        A_i = C @ np.linalg.pinv(V)

        # QR factorization: A_i * Q_i = Q_{i+1} * R_i
        Q, R = np.linalg.qr(A_i)
        # TODO: CHECK THIS. This doesn't necessary calc max lyapunov exponents
        local_lyap = (1 / (m * tau)) * np.max(np.log(np.abs(np.diag(R))))
        lyap_timeseries.append(local_lyap)
    else:
        lyap_timeseries.append(np.full(d_E, np.nan))  # Pad with NaNs if not enough neighbors

# === Convert to array for plotting and analysis ===
lyap_timeseries = np.array(lyap_timeseries)

# === Plot max Lyapunov exponent over time ===
plt.figure(figsize=(12, 4))
# DOES NOT MAKE SENSE
plt.plot(lyap_timeseries, label="λ₁ (Max Lyapunov)")
plt.xlabel("Time step")
plt.ylabel("Local Lyapunov Exponent")
plt.title("Local Lyapunov Exponent Time Series (λ₁)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
