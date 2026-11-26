
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load data
df = pd.read_csv("data/household_debt_income_2024_verified.csv")

features = [
    "household_debt_2024_03_만원",
    "household_income_2023_만원"
]

X = df[features].dropna()

# 2. Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Compare different numbers of clusters
results = []

for k in range(2, 7):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    results.append({
        "k": k,
        "inertia": inertia,
        "silhouette": silhouette
    })

results_df = pd.DataFrame(results)

print("=== K-Means comparison ===")
print(results_df.to_string(index=False))

# 4. Elbow method
plt.figure(figsize=(8, 5))
plt.plot(
    results_df["k"],
    results_df["inertia"],
    marker="o"
)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# 5. Silhouette comparison
plt.figure(figsize=(8, 5))
plt.plot(
    results_df["k"],
    results_df["silhouette"],
    marker="o"
)
plt.title("Silhouette Score Comparison")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()
