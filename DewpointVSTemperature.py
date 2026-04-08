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


# Load & Clean Data

appliances_energy_prediction = fetch_ucirepo(id=374) 

X = appliances_energy_prediction.data.features 
y = appliances_energy_prediction.data.targets 

data = pd.concat([X, y], axis=1)
clean_data = data.iloc[:1500][["T_out", "Tdewpoint"]]

z_scores = (clean_data - clean_data.mean()) / clean_data.std()
clean_data = clean_data[(np.abs(z_scores) < 1.5).all(axis=1)]

X = clean_data[["Tdewpoint"]].values
y = clean_data["T_out"].values


# Split into train & test

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Pytorch Model

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


# Train MLP

def train_mlp(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)
    X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)

    model = MLP()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        model.train()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test_scaled).numpy().flatten()

    return preds

def train_mlp_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)

    model = MLP()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, scaler

def predict_mlp(model, scaler, X):
    X_scaled = scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        return model(X_t).numpy().flatten()


# Linear Regression

def linear_regression_predict(X_train, y_train, X_test):
    slope, intercept = np.polyfit(X_train.flatten(), y_train, 1)
    return slope * X_test.flatten() + intercept


# Experiment for each fraction of dataset

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

            # Linear Regression
            y_pred_lin = linear_regression_predict(X_train, y_train, X_test)
            lin_mse = mean_squared_error(y_test, y_pred_lin)
            lin_errors.append(lin_mse)

            # PyTorch ANN
            y_pred_ann = train_mlp(X_train, y_train, X_test)
            ann_mse = mean_squared_error(y_test, y_pred_ann)
            ann_errors.append(ann_mse)

        results[frac] = {
            "Linear Regression": (np.mean(lin_errors), np.std(lin_errors)),
            "PyTorch MLP": (np.mean(ann_errors), np.std(ann_errors))
        }
    return results


# Run Experiment and Print Results

results = run_experiment()

for frac, res in results.items():
    print(f"\nTraining Size: {int(frac*100)}%")
    for model, (mean, std) in res.items():
        print(f"{model}: {mean:.4f} ± {std:.4f}")

# Graph results for Lin Reg

x_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

plt.figure()

# plot ALL data points
plt.scatter(X, y, alpha=0.3)

for frac in [0.1, 0.3, 0.5, 1.0]:
    idx = np.random.choice(len(X_train_full),
                           int(len(X_train_full)*frac),
                           replace=False)

    X_train = X_train_full[idx]
    y_train = y_train_full[idx]

    slope, intercept = np.polyfit(X_train.flatten(), y_train, 1)

    y_line = slope * x_plot.flatten() + intercept

    plt.plot(x_plot, y_line, label=f"{int(frac*100)}%")

plt.xlabel("Temperature Outside")
plt.ylabel("Dewpoint")
plt.title("Linear Regression (Different Training Sizes)")
plt.legend()
plt.show()

# Plot results for ANN

# plot ALL data points
plt.figure()

# plot ALL data points
plt.scatter(X, y, alpha=0.3)

for frac in [0.1, 0.3, 0.5, 1.0]:
    idx = np.random.choice(len(X_train_full),
                           int(len(X_train_full)*frac),
                           replace=False)

    X_train = X_train_full[idx]
    y_train = y_train_full[idx]

    # train model properly
    model, scaler = train_mlp_model(X_train, y_train)

    # generate smooth curve
    y_curve = predict_mlp(model, scaler, x_plot)

    plt.plot(x_plot, y_curve, label=f"{int(frac*100)}%")

plt.xlabel("Temperature Outside")
plt.ylabel("Dewpoint")
plt.title("ANN (MLP) (Different Training Sizes)")
plt.legend()
plt.show()