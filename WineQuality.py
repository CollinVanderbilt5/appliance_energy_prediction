from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# Load Data
wine_quality = fetch_ucirepo(id=186)

X = wine_quality.data.features 
y = wine_quality.data.targets 

data = pd.concat([X, y], axis=1)

clean_data = data.iloc[:1500][["free_sulfur_dioxide", "total_sulfur_dioxide"]]

z_scores = (clean_data - clean_data.mean()) / clean_data.std()
clean_data = clean_data[(np.abs(z_scores) < 1.5).all(axis=1)]

X = clean_data[["free_sulfur_dioxide"]].values
y = clean_data["total_sulfur_dioxide"].values


# Train/Test Split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# MLP Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


# Fixed MLP Training
def train_mlp(X_train, y_train, X_test):
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1,1))

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    model = MLP()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(300):
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_t).numpy()
        preds = y_scaler.inverse_transform(preds_scaled).flatten()

    return preds


# Linear Regression
def linear_regression_predict(X_train, y_train, X_test):
    slope, intercept = np.polyfit(X_train.flatten(), y_train, 1)
    return slope * X_test.flatten() + intercept


# Experiment
fractions = [0.1, 0.3, 0.5, 1.0]

def run_experiment():
    results = {}

    for frac in fractions:
        lin_errors = []
        ann_errors = []

        for _ in range(3):
            idx = np.random.choice(len(X_train_full),
                                   int(len(X_train_full)*frac),
                                   replace=False)

            X_train = X_train_full[idx]
            y_train = y_train_full[idx]

            y_pred_lin = linear_regression_predict(X_train, y_train, X_test)
            lin_mse = mean_squared_error(y_test, y_pred_lin)
            lin_errors.append(lin_mse)

            y_pred_ann = train_mlp(X_train, y_train, X_test)
            ann_mse = mean_squared_error(y_test, y_pred_ann)
            ann_errors.append(ann_mse)

        results[frac] = {
            "Linear Regression": (np.mean(lin_errors), np.std(lin_errors)),
            "PyTorch MLP": (np.mean(ann_errors), np.std(ann_errors))
        }

    return results


# Run + Print
results = run_experiment()

for frac, res in results.items():
    print(f"\nTraining Size: {int(frac*100)}%")
    for model, (mean, std) in res.items():
        print(f"{model}: {mean:.4f} ± {std:.4f}")


# Plot Linear Regression
x_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

plt.figure()
plt.scatter(X, y, alpha=0.3)

for frac in fractions:
    idx = np.random.choice(len(X_train_full),
                           int(len(X_train_full)*frac),
                           replace=False)

    X_train = X_train_full[idx]
    y_train = y_train_full[idx]

    slope, intercept = np.polyfit(X_train.flatten(), y_train, 1)
    y_line = slope * x_plot.flatten() + intercept

    plt.plot(x_plot, y_line, label=f"{int(frac*100)}%")

plt.xlabel("Free Sulfur Dioxide")
plt.ylabel("Total Sulfur Dioxide")
plt.title("Linear Regression")
plt.legend()
plt.show()


# Plot MLP
plt.figure()
plt.scatter(X, y, alpha=0.3)

for frac in fractions:
    idx = np.random.choice(len(X_train_full),
                           int(len(X_train_full)*frac),
                           replace=False)

    X_train = X_train_full[idx]
    y_train = y_train_full[idx]

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1,1))

    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)

    for epoch in range(300):
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    X_plot_scaled = X_scaler.transform(x_plot)
    X_plot_t = torch.tensor(X_plot_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_curve_scaled = model(X_plot_t).numpy()
        y_curve = y_scaler.inverse_transform(y_curve_scaled)

    plt.plot(x_plot, y_curve, label=f"{int(frac*100)}%")

plt.xlabel("Free Sulfur Dioxide")
plt.ylabel("Total Sulfur Dioxide")
plt.title("MLP Regression")
plt.legend()
plt.show()