# Appliance Energy Usage Prediction & Benchmarking Pipeline

by Colin Vanderbilt, Dorian Williams-Webster, Haley Hankins, Will Dorsey

## Overview
This repository contains a end-to-end machine learning pipeline built in Python to predict household energy consumption using sensor data from the UCI Machine Learning Repository. 

The core goal of the project was to design a modular, reproducible evaluation framework to benchmark deep learning models against traditional regression baselines across varying training set sizes.

## Key Highlights
* **Deep Architecture:** Designed and trained a **Multi-Layer Perceptron (MLP)** in PyTorch using Mean Squared Error (MSE) loss across 200 epochs.
* **Baseline Benchmarking:** Evaluated PyTorch neural network performance against baseline **Scikit-Learn Linear Regression** models.
* **Evaluation & Metrics:** Analyzed convergence, generalization, and error rates across fractional training splits using Matplotlib learning curves.
* **Modular Design:** Built as a reusable pipeline structure applicable across multiple tabular datasets.

## Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch
* **Machine Learning & Preprocessing:** Scikit-Learn, Pandas
* **Visualization:** Matplotlib
  
## Installation

Clone this repo
To access the virtual enviorment enter the following command:

    source venv/bin/activate

Then with the virtual enviorment activated, to get the required libraries enter this command

    pip install -r requirements.txt
