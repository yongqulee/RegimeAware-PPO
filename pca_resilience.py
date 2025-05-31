import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

df = pd.read_csv('asset_resilience_scores.csv')
features = ['Vol_inv', 'Cumulative Return', 'CAGR']
X = StandardScaler().fit_transform(df[features])

pca = PCA()
pca_resilience = pca.fit_transform(X)[:, 0]

df['PCA_Resilience'] = MinMaxScaler().fit_transform(pca_resilience.reshape(-1,1))
df.to_csv('pca_asset_resilience.csv', index=False)
