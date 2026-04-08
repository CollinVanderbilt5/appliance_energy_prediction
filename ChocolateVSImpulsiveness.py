from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# fetch dataset
drug_consumption_quantified = fetch_ucirepo(id=373)

# data (as pandas dataframes)
X = drug_consumption_quantified.data.features
y = drug_consumption_quantified.data.targets

#collect first 1500 rows of dataset
data = pd.concat([X, y], axis=1)
cleanData = data.iloc[:1500][["impulsive", "choc"]]

avgImpulseScore = cleanData["impulsive"].mean()
avgChocConsumption = cleanData["choc"].mean()
z_scores = (cleanData - cleanData.mean()) / cleanData.std()

finalCleanData = cleanData[(np.abs(z_scores)<1.5).all(axis=1)]

def print_set(dataset):
    plt.scatter(dataset["choc"], dataset["impulsive"])
    plt.xlabel("Chocolate Consumption")
    plt.ylabel("Impulsivity Score")
    plt.title("Chocolate Consumption vs Impulsivity Score")
    plt.show()

def run_trial():
    #10% of the dataset
    sample10 - finalCleanData.sample(frac=0.1)
    slope10, intercept10 = np.polyfit(
        sample10["impulsive"]
        sample_10["choc"]
        1
    )
    print(f"Slope for 10% sample: {slope10}")

    #30% of the dataset
    sample_30 = finalCleanData.sample(frac=0.3)
    slope30, intercept30 = np.polyfit(
        sample30["impulsive"]
        sample30["choc"]
        1
    )
    print(f"Slope for 30% sample: {slope30}")

    #50% of the dataset
    sample50 = finalCleanData.sample(frac=0.5)
    slope50, intercept50 = np.polyfit(
        sample50["impulsive"]
        sample50["choc"]
        1
    )
    print(f"Slope for 50% sample: {slope50}")

    #100% of the dataset
    sample100 = finalCleanData
    slope100, intercept100 = np.polyfit(
        sample100["impulse"],
        sample100["choc"],
        1
    )

    #compare Linear regression lines
    #slopes
    s10, b10 = np.polyfit(sample10["impulsive"], sample10["choc"], 1)
    s30, b30 = np.polyfit(sample_30["Tdewpoint"], sample_30["T_out"], 1)
    s50, b50 = np.polyfit(sample_50["Tdewpoint"], sample_50["T_out"], 1)
    s100, b100 = np.polyfit(sample_100["Tdewpoint"], sample_100["T_out"], 1)

    x = clean_data["impulsive"]

    plt.scatter(sample_100["impulsive"], sample_100["choc"], alpha=0.5)

    plt.plot(x, s10*x + b10, label=f"10% Sample")
    plt.plot(x, s30*x + b30, label=f"30% Sample")
    plt.plot(x, s50*x + b50, label=f"50% Sample")
    plt.plot(x, s100*x + b100, label=f"100% Sample")

    plt.legend()
    plt.show()

# run 3 trials to compare results
print("Trial one:")
run_trial()
print("\nTrial two:")
run_trial()
print("\nTrial three:")
run_trial()
