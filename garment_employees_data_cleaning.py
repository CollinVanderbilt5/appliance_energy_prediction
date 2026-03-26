from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
  
# fetch dataset 
productivity_prediction_of_garment_employees = fetch_ucirepo(id=597) 
# data (as pandas dataframes) 
X = productivity_prediction_of_garment_employees.data.features 
y = productivity_prediction_of_garment_employees.data.targets 

# size of dataset
print(f"Size of dataset: {X.shape[0]}, with {X.shape[1]} features")

# Quality Checks
print("Missing values in each column:\n", X.isnull().sum())

# Drop the 'wip' column because a lot of it is missing
X = X.drop(columns=['wip'])

# Record the original count before finding outliers to remove
original_count = len(X)

# use z values for outlier checks
z_scores = (X['no_of_workers'] - X['no_of_workers'].mean()) / X['no_of_workers'].std()

# for X and y, keep only rows less than 1.5 std's away
X_clean = X[z_scores.abs() < 1.5]
y_clean = y[z_scores.abs() < 1.5]

# Print results
removed_count = original_count - len(X_clean)
print(f"Number of original samples: {original_count}")
print(f"Number of clean samples: {len(X_clean)}")
print(f"Removed outliers: {removed_count}")

# Scaling, using the min and max from the entire X_clean
w_min = X_clean['no_of_workers'].min()
w_max = X_clean['no_of_workers'].max()

# Create the scaled column directly in X_clean
X_clean['workers_scaled'] = (X_clean['no_of_workers'] - w_min) / (w_max - w_min)

# Splitting data (first shuffling to ensure randomness)
df_final = pd.concat([X_clean, y_clean], axis=1).sample(frac=1, random_state=26)

train_10 = df_final.sample(frac=0.1, random_state=42)
train_30 = df_final.sample(frac=0.3, random_state=42)
train_50 = df_final.sample(frac=0.5, random_state=42)
train_100 = df_final # 100% of cleaned data

testing_dataset = df_final

print(f"10% size: {len(train_10)}")
print(f"30% size: {len(train_30)}")
print(f"50% size: {len(train_50)}")
print(f"100% size: {len(train_100)}")

print(f"Testing dataset size: {len(testing_dataset)}")

# Create a figure with two side-by-side plots
plt.figure(figsize=(10, 5))

# 1st graph
plt.subplot(1, 2, 1) # 1 row, 2 columns, index 1
plt.scatter(X['no_of_workers'], y['actual_productivity'], alpha=0.5, color='blue')
plt.title('Workers vs Productivity')
plt.xlabel('Number of Workers')
plt.ylabel('Actual Productivity')
plt.grid(True)

# 2nd graph (cleaned)
plt.subplot(1, 2, 2) # 1 row, 2 columns, index 2
plt.scatter(X_clean['no_of_workers'], y_clean['actual_productivity'], alpha=0.5, color='green')
plt.title('CLEANED Workers vs Productivity')
plt.xlabel('Number of Workers')
plt.ylabel('Actual Productivity')
plt.grid(True)

plt.tight_layout()
plt.show()

# SOME NOTES:

# I realize actual_productivity likely scales with team size, however it is still worth
# observing whether smaller teams tend to meet their goals more or less than larger ones

# We are pulling our data splits from the original data, and thus our 100%
# split will be the SAME as the data that training is tested against. For now, since the
# instructions don't mention anything else, we will proceed with this