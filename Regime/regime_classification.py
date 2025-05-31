import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

df = pd.read_csv("regime_features.csv")
X = df.drop(columns=["Year"])
y = df["Year"].isin([1931, 1974, 1987, 2001, 2008, 2020]).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Autoencoder
input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation='relu')(input_layer)
encoded = Dense(3, activation='relu')(encoded)
decoded = Dense(8, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=8, shuffle=True, verbose=0)

encoder = Model(input_layer, encoded)
X_latent = encoder.predict(X_scaled)

# Random Forest & XGBoost
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)

print(classification_report(y_test, rf.predict(X_test)))
print(confusion_matrix(y_test, rf.predict(X_test)))
print("ROC AUC RF:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))

print(classification_report(y_test, xgb.predict(X_test)))
print(confusion_matrix(y_test, xgb.predict(X_test)))
print("ROC AUC XGB:", roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]))
