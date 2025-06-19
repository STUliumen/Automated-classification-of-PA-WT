import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('features/features_0.csv')

X = df.iloc[:, 2:]
y = df['class']

y = y.map({'PA': 0, 'WT': 1})


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

num_features_selected = []
alphas = np.logspace(-6, 0, 50)


lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)


selected_features = X.columns[lasso.coef_ != 0]
selected_features_df = pd.DataFrame({df.columns[0]: df.iloc[:, 0], **dict(zip(selected_features, X[selected_features].T.values))})

selected_features_df.to_csv('features/lasso/lasso_0.csv', index=False)