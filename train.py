import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Generate customer data (Annual Income in $k, Spending Score 1-100)
np.random.seed(42)

cluster1 = np.random.normal(loc=[25, 20], scale=[5, 5], size=(100, 2))  # Low Income, Low Spending (Sensible)
cluster2 = np.random.normal(loc=[25, 80], scale=[5, 5], size=(100, 2))  # Low Income, High Spending (Careless)
cluster3 = np.random.normal(loc=[85, 20], scale=[5, 5], size=(100, 2))  # High Income, Low Spending (Careful)
cluster4 = np.random.normal(loc=[85, 85], scale=[5, 5], size=(100, 2))  # High Income, High Spending (VIP/Target)

X = np.vstack([cluster1, cluster2, cluster3, cluster4])

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit KMeans (4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Ensure api folder exists and dump artifacts
os.makedirs("api", exist_ok=True)
joblib.dump(kmeans, "api/model.pkl")
joblib.dump(scaler, "api/scaler.pkl")

print("--> SUCCESS: model.pkl and scaler.pkl created inside /api directory!")