
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# fetch dataset 
appliances_energy_prediction = fetch_ucirepo(id=374) 
  
# target (y) = # appliances (int)
X = appliances_energy_prediction.data.features 
y = appliances_energy_prediction.data.targets 

X = X.drop(columns=['date', 'rv1', 'rv2'])

X.dropna(inplace=True)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_r2s = []
test_r2s = []

def run_linear_trial(size):
  scaler = StandardScaler()

  X_train = X_train_full.sample(frac=size, random_state=42)
  y_train = y_train_full.loc[X_train.index]

  X_train = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  linear_model = LinearRegression()

  linear_model.fit(X_train, y_train)

  y_train_predict = linear_model.predict(X_train)
  y_test_pred = linear_model.predict(X_test_scaled)

  train_r2 = r2_score(y_train, y_train_predict)
  test_r2 = r2_score(y_test, y_test_pred)

  print(f"Linear R2 score (train): {train_r2:.4f}")
  print(f"LinearR2 score (test): {test_r2:.4f}")

  train_r2s.append(train_r2)
  test_r2s.append(test_r2)

fractions = [0.1, 0.3, 0.5, 1.0]
for frac in fractions:
  run_linear_trial(frac)

plt.plot(fractions, train_r2s, label="Train R²")
plt.plot(fractions, test_r2s, label="Test R²")
plt.xlabel("Training Set Fraction")
plt.ylabel("R² Score")
plt.legend()
plt.title("Learning Curve (Linear Regression)")
plt.show()


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)
    

train_r2s = []
test_r2s = []

def run_mlp_trial(size):
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = X_train_full.sample(frac=size, random_state=42)
    y_train = y_train_full.loc[X_train.index]

    X_train = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    y_train_scaled = y_scaler.fit_transform(y_train.values)
    y_test_scaled = y_scaler.transform(y_test.values)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    model = MLP(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()

        preds = model(X_train_tensor)
        loss = loss_fn(preds, y_train_tensor)

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_pred_scaled = model(X_train_tensor).numpy()
        test_pred_scaled = model(X_test_tensor).numpy()

    
    train_pred = y_scaler.inverse_transform(train_pred_scaled)
    test_pred = y_scaler.inverse_transform(test_pred_scaled)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    print(f"MLP R2 score (train): {train_r2:.4f}")
    print(f"MLP R2 score (test): {test_r2:.4f}")

    train_r2s.append(train_r2)
    test_r2s.append(test_r2)

for frac in fractions:
    run_mlp_trial(frac)

plt.plot(fractions, train_r2s, label="Train R²")
plt.plot(fractions, test_r2s, label="Test R²")
plt.xlabel("Training Set Fraction")
plt.ylabel("R² Score")
plt.legend()
plt.title("Learning Curve (Multi-Layer Perceptron)")
plt.show()
