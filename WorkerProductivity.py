from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# fetch dataset 
productivity_prediction_of_garment_employees = fetch_ucirepo(id=597) 
# data (as pandas dataframes) 
X = productivity_prediction_of_garment_employees.data.features 
y = productivity_prediction_of_garment_employees.data.targets 

data = pd.concat([X, y], axis=1)

clean_data = data.iloc[:1500][["no_of_workers", "actual_productivity"]]

z_scores = (clean_data - clean_data.mean()) / clean_data.std()
clean_data = clean_data[(np.abs(z_scores) < 1.5).all(axis=1)]

X = clean_data[["no_of_workers"]].values
y = clean_data["actual_productivity"].values



# Split into train & test: COMPLETE!! Note: Testing 20%, Training 80%

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


def train_mlp(X_train, y_train, epochs=300):
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1,1))

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)

    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, X_scaler, y_scaler

def predict_mlp(model, X_scaler, y_scaler, X):
    X_scaled = X_scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_scaled = model(X_t).numpy()

    return y_scaler.inverse_transform(y_scaled).flatten()

def train_linear(X_train, y_train):
    slope, intercept = np.polyfit(X_train.flatten(), y_train, 1)
    return slope, intercept

def predict_linear(slope, intercept, X):
    return slope * X.flatten() + intercept

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

def run_experiment():
    results = {}
    fractions = [0.1, 0.3, 0.5, 1.0]

    for frac in fractions:
        lin_train_err, lin_test_err = [], []
        ann_train_err, ann_test_err = [], []

        for _ in range(3):
            idx = np.random.choice(len(X_train_full),
                                   int(len(X_train_full)*frac),
                                   replace=False)

            X_train = X_train_full[idx]
            y_train = y_train_full[idx]

            # Lin Reg
            slope, intercept = train_linear(X_train, y_train)

            y_train_pred = predict_linear(slope, intercept, X_train)
            y_test_pred  = predict_linear(slope, intercept, X_test)

            lin_train_err.append(mean_squared_error(y_train, y_train_pred))
            lin_test_err.append(mean_squared_error(y_test, y_test_pred))

            # ANN
            model, x_scaler, y_scaler = train_mlp(X_train, y_train)

            y_train_pred = predict_mlp(model, x_scaler, y_scaler, X_train)
            y_test_pred  = predict_mlp(model, x_scaler, y_scaler, X_test)

            ann_train_err.append(mean_squared_error(y_train, y_train_pred))
            ann_test_err.append(mean_squared_error(y_test, y_test_pred))

        results[frac] = {
            "Linear": {
                "train": np.mean(lin_train_err),
                "test": np.mean(lin_test_err)
            },
            "ANN": {
                "train": np.mean(ann_train_err),
                "test": np.mean(ann_test_err)
            }
        }

    return results

results = run_experiment()

for frac, res in results.items():
    print(f"\nTraining Size: {int(frac*100)}%")

    for model in ["Linear", "ANN"]:
        train_err = res[model]["train"]
        test_err = res[model]["test"]
        gap = test_err - train_err

        print(f"{model}:")
        print(f"  Train MSE: {train_err:.4f}")
        print(f"  Test  MSE: {test_err:.4f}")
        print(f"  Gap       : {gap:.4f}")

fractions = [0.1, 0.3, 0.5, 1.0]

lin_train = [results[f]["Linear"]["train"] for f in fractions]
ann_train = [results[f]["ANN"]["train"] for f in fractions]

plt.figure()
plt.plot(fractions, lin_train, marker='o', label="Linear")
plt.plot(fractions, ann_train, marker='o', label="ANN")

plt.xlabel("Training Data Fraction")
plt.ylabel("Training MSE")
plt.title("Training Error vs Data Size")
plt.legend()
plt.show()

lin_test = [results[f]["Linear"]["test"] for f in fractions]
ann_test = [results[f]["ANN"]["test"] for f in fractions]

plt.figure()
plt.plot(fractions, lin_test, marker='o', label="Linear")
plt.plot(fractions, ann_test, marker='o', label="ANN")

plt.xlabel("Training Data Fraction")
plt.ylabel("Test MSE")
plt.title("Test Error vs Data Size")
plt.legend()
plt.show()

lin_gap = [results[f]["Linear"]["test"] - results[f]["Linear"]["train"] for f in fractions]
ann_gap = [results[f]["ANN"]["test"] - results[f]["ANN"]["train"] for f in fractions]

plt.figure()
plt.plot(fractions, lin_gap, marker='o', label="Linear")
plt.plot(fractions, ann_gap, marker='o', label="ANN")

plt.xlabel("Training Data Fraction")
plt.ylabel("Generalization Gap")
plt.title("Overfitting vs Data Size")
plt.legend()
plt.show()