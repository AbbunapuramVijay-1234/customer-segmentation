import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# 1. Automatically fetch the real Mall_Customers.csv dataset and save locally
url = "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv"
df = pd.read_csv(url)

# Save a physical copy in your project folder so you can view it in VS Code!
df.to_csv("Mall_Customers.csv", index=False)
print("--> Downloaded and saved Mall_Customers.csv locally!")

# 2. Select Annual Income ($k) and Spending Score (1-100)
X = df.iloc[:, [3, 4]].values

# 3. Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train K-Means (5 standard clusters for Mall Customers)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# 5. Save model and scaler
os.makedirs("api", exist_ok=True)
joblib.dump(kmeans, "api/model.pkl")
joblib.dump(scaler, "api/scaler.pkl")

print("--> SUCCESS: Trained on real Mall_Customers.csv and saved model.pkl!")