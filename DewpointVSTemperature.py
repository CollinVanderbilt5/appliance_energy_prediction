from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# fetch dataset 
appliances_energy_prediction = fetch_ucirepo(id=374) 
  
# data (as pandas dataframes) 
X = appliances_energy_prediction.data.features 
y = appliances_energy_prediction.data.targets 
  
# metadata 
# print(appliances_energy_prediction.metadata) 
  
# variable information 
# print(appliances_energy_prediction.variables) 

# collect first 1500 rows of dataset
data = pd.concat([X, y], axis=1)
clean_data = data.iloc[:1500][["T_out", "Tdewpoint"]]

avg_temp = clean_data["T_out"].mean()
avg_dew = clean_data["Tdewpoint"].mean()

z_scores = (clean_data - clean_data.mean()) / clean_data.std()

clean_data_no_outliers = clean_data[(np.abs(z_scores) < 1.5).all(axis=1)]

print(clean_data_no_outliers.mean())

print("Original rows:", len(clean_data))
print("After outlier removal:", len(clean_data_no_outliers))



plt.scatter(clean_data_no_outliers["Tdewpoint"], clean_data_no_outliers["T_out"])
plt.xlabel("Dewpoint")
plt.ylabel("Temperature Outside")
plt.title("Dewpoint vs Temperature Outside")
plt.show()