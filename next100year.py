import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool

# 1. Load Dataset
data = pd.read_excel("dataset.xlsx")  # Replace 'dataset.xlsx' with your file name

print("Columns in DataFrame:", data.columns)
print("First few rows:")
print(data.head())

# 2. Split Features and Target
X = data.drop(columns=["flow_rate", "station"])
Y = data["flow_rate"]

# Include the year variable in modeling
X["year"] = data["year"]

# 3. Define Numerical and Categorical Features
numerical_features = ["precipitation", "temperature", "slope", "basin_area", "year", "elevation", "main_stream"]
categorical_features = ["land_use", "aspect", "awc_top", "oc_top"]
all_features = numerical_features + categorical_features

# 4. Adjust Data Types of Categorical Features
X[categorical_features] = X[categorical_features].astype(str)

# 5. Split Data into Train and Test Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 6. Create Data Pool for CatBoost
train_pool = Pool(X_train, Y_train, cat_features=categorical_features)
test_pool = Pool(X_test, Y_test, cat_features=categorical_features)

# 7. Train CatBoost Model
print("\nTraining CatBoost Model...")
cat_model = CatBoostRegressor(
    iterations=1000,
    depth=10,
    learning_rate=0.03,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=100
)
cat_model.fit(train_pool)

# 8. Get Feature Importances
feature_importances = cat_model.get_feature_importance()
feature_importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)
print("\nFeature Importances:")
print(feature_importance_df)

# 9. Prepare Data for Future Years
future_years = np.arange(2010, 2110, 5)  # From 2010 to 2105, every 5 years

if "station" not in data.columns:
    print("Error: 'station' column is missing!")
    print("Available columns:", data.columns)
    raise KeyError("'station' column is missing!")
else:
    print("The 'station' column exists and contains the following values:")
    print(data["station"].unique())

# Create Station and Year Data
stations = data["station"].unique()
future_data = pd.DataFrame({
    "year": np.tile(future_years, len(stations)),
    "station": np.repeat(stations, len(future_years))
})

print("Initial Future Data:")
print(future_data.head())

# Add changes in features over time and by station
station_means = data.groupby("station")[numerical_features].mean()  # Numerical columns only
station_modes = data.groupby("station")[categorical_features].agg(lambda x: x.mode()[0])  # Categorical columns

future_data = future_data.merge(station_means.reset_index(), on="station", how="left")
future_data = future_data.merge(station_modes.reset_index(), on="station", how="left")

# Check for missing 'year' column and re-add if necessary
if "year" not in future_data.columns:
    future_data["year"] = np.tile(future_years, len(stations))

future_data["temperature"] += 0.2 * ((future_data["year"] - 2010) // 5)
future_data["precipitation"] *= (1 + 0.01 * ((future_data["year"] - 2010) // 5))

# Remove duplicate 'station' columns
future_data = future_data.loc[:, ~future_data.columns.duplicated()]

# Match column order with training data
future_data = future_data[[col for col in X_train.columns if col in future_data.columns] + ["station"]]

# 10. Make Predictions for Future Data
print("Future data before creating Pool:")
print(future_data.head())

future_pool = Pool(future_data.drop(columns=["station"]), cat_features=categorical_features)

try:
    future_data["predicted_flow_rate"] = cat_model.predict(future_pool)
except Exception as e:
    print("An error occurred during prediction:", str(e))
    raise

# Check if predictions were added
if "predicted_flow_rate" not in future_data.columns:
    print("Error: 'predicted_flow_rate' column was not added after prediction.")
    raise KeyError("'predicted_flow_rate' column is missing.")

# Final flow rate calculation (flow rate + precipitation)
future_data["final_flow_rate"] = future_data["predicted_flow_rate"] + future_data["precipitation"]

print("Future data after adding predictions and final flow rates:")
print(future_data.head())

# 11. Station-Wise and Yearly Average Predictions
if "station" not in future_data.columns or "year" not in future_data.columns:
    print("Error: Required columns for groupby (station or year) are missing.")
    raise KeyError("Required columns for groupby are missing.")

station_yearly_averages = future_data.groupby(["station", "year"])["final_flow_rate"].mean().reset_index()
print("Groupby operation completed.")

# 12. Visualization: Station-Wise Predictions
plt.figure(figsize=(12, 6))
for station in station_yearly_averages["station"].unique():
    station_data = station_yearly_averages[station_yearly_averages["station"] == station]
    plt.plot(
        station_data["year"],
        station_data["final_flow_rate"],
        label=f"Station {station}"
    )

plt.title("Final Flow Rate Over Future Years (Station-wise Averages)")
plt.xlabel("Year")
plt.ylabel("Average Final Flow Rate")
plt.legend(title="Stations", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()

# 13. Save Future Predictions
future_data.to_excel("future_flow_rate_predictions.xlsx", index=False)
station_yearly_averages.to_excel("station_yearly_averages.xlsx", index=False)
print("\nFuture predictions saved to 'future_flow_rate_predictions.xlsx'.")
print("Station-wise yearly averages saved to 'station_yearly_averages.xlsx'.")

