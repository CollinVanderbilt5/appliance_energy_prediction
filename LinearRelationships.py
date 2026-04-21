# This file outputs graphs using matplotlib to show a linear correlation between two variables in each dataset

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function to output graph

def graph(x, y, x_label, y_label, title):
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    s, b = np.polyfit(x, y, 1)
    plt.plot(x, s*x + b)
    plt.show()




# Energy Prediction

appliances_energy_prediction = fetch_ucirepo(id=374) 

X = appliances_energy_prediction.data.features 
y = appliances_energy_prediction.data.targets 

data = pd.concat([X, y], axis=1)
clean_data = data.iloc[:1500][["T_out", "Tdewpoint"]]

z_scores = (clean_data - clean_data.mean()) / clean_data.std()
clean_data = clean_data[(np.abs(z_scores) < 3).all(axis=1)]

X = clean_data["T_out"]
y = clean_data["Tdewpoint"]

graph(X, y, "Temperature Outside", "Dewpoint", "Energy Prediction Dataset")


# Wine Quality

wine_quality = fetch_ucirepo(id=186)

X = wine_quality.data.features 
y = wine_quality.data.targets 

data = pd.concat([X, y], axis=1)

clean_data = data.iloc[:1500][["free_sulfur_dioxide", "total_sulfur_dioxide"]]

z_scores = (clean_data - clean_data.mean()) / clean_data.std()
clean_data = clean_data[(np.abs(z_scores) < 1.5).all(axis=1)]

X = clean_data["free_sulfur_dioxide"]
y = clean_data["total_sulfur_dioxide"]

graph(X, y, "Free Sulfur Dioxide", "Total Sulfur Dioxide", "Wine Quality Dataset")


# Worker Productivity

productivity_prediction_of_garment_employees = fetch_ucirepo(id=597) 

X = productivity_prediction_of_garment_employees.data.features 
y = productivity_prediction_of_garment_employees.data.targets 

data = pd.concat([X, y], axis=1)

clean_data = data.iloc[:1500][["no_of_workers", "actual_productivity"]]

z_scores = (clean_data - clean_data.mean()) / clean_data.std()
clean_data = clean_data[(np.abs(z_scores) < 1.5).all(axis=1)]

X = clean_data["no_of_workers"]
y = clean_data["actual_productivity"]

graph(X, y, "Number of Workers", "Productivity", "Productivity Prediction Dataset")
