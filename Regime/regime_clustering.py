import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

df = pd.read_csv("regime_features.csv")
X = df.drop(columns=["Year"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Regime_KMeans'] = kmeans.fit_predict(X_scaled)

gmm = GaussianMixture(n_components=3, random_state=42)
df['Regime_GMM'] = gmm.fit_predict(X_scaled)

hmm = GaussianHMM(n_components=3, covariance_type='full', n_iter=1000, random_state=42)
hmm.fit(X_scaled)
df['Regime_HMM'] = hmm.predict(X_scaled)

df.to_csv("regime_clusters_output.csv", index=False)

plt.plot(df["Year"], df["Regime_HMM"], label="HMM")
plt.plot(df["Year"], df["Regime_KMeans"], label="KMeans", linestyle="--")
plt.plot(df["Year"], df["Regime_GMM"], label="GMM", linestyle=":")
plt.legend()
plt.title("Regime Clustering")
plt.savefig("regime_comparison_plot.png")
plt.show()
