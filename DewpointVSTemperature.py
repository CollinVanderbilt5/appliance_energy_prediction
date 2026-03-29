from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# fetch dataset 
appliances_energy_prediction = fetch_ucirepo(id=374) 
  
# data (as pandas dataframes) 
X = appliances_energy_prediction.data.features 
y = appliances_energy_prediction.data.targets 

# collect first 1500 rows of dataset
data = pd.concat([X, y], axis=1)
clean_data = data.iloc[:1500][["T_out", "Tdewpoint"]]

avg_temp = clean_data["T_out"].mean()
avg_dew = clean_data["Tdewpoint"].mean()
z_scores = (clean_data - clean_data.mean()) / clean_data.std()

clean_data_no_outliers = clean_data[(np.abs(z_scores) < 1.5).all(axis=1)]


# print("Original rows:", len(clean_data))
# print("After outlier removal:", len(clean_data_no_outliers))


# plot data set: dewpoint vs temp outside
def print_set(dataset):
    plt.scatter(dataset["Tdewpoint"], dataset["T_out"])
    plt.xlabel("Dewpoint")
    plt.ylabel("Temperature Outside")
    plt.title("Dewpoint vs Temperature Outside")
    plt.show()

def run_trial():
    # take 10% sample of dataset
    sample_10 = clean_data_no_outliers.sample(frac=0.1)
    slope_10, intercept_10 = np.polyfit(
        sample_10["Tdewpoint"],
        sample_10["T_out"],
        1
    )
    print(f"Slope for 10% sample: {slope_10}")

    # take 30% sample of dataset
    sample_30 = clean_data_no_outliers.sample(frac=0.3)
    slope_30, intercept_30 = np.polyfit(
        sample_30["Tdewpoint"],
        sample_30["T_out"],
        1
    )
    print(f"Slope for 30% sample: {slope_30}")

    # take 50% sample of dataset
    sample_50 = clean_data_no_outliers.sample(frac=0.5)
    slope_50, intercept_50 = np.polyfit(
        sample_50["Tdewpoint"],
        sample_50["T_out"],
        1
    )
    print(f"Slope for 50% sample: {slope_50}")

    # take 100% sample of dataset (entire set after cleaning)
    sample_100 = clean_data_no_outliers
    slope_100, intercept_100 = np.polyfit(
        sample_100["Tdewpoint"],
        sample_100["T_out"],
        1
    )
    print(f"Slope for 100% sample: {slope_100}")


    # compare linear regression lines
    # slopes
    s10, b10 = np.polyfit(sample_10["Tdewpoint"], sample_10["T_out"], 1)
    s30, b30 = np.polyfit(sample_30["Tdewpoint"], sample_30["T_out"], 1)
    s50, b50 = np.polyfit(sample_50["Tdewpoint"], sample_50["T_out"], 1)
    s100, b100 = np.polyfit(sample_100["Tdewpoint"], sample_100["T_out"], 1)

    x = clean_data["Tdewpoint"]

    plt.scatter(sample_100["Tdewpoint"], sample_100["T_out"], alpha=0.5)

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