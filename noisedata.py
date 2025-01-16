import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool

# 1. Load Dataset
data = pd.read_excel("dataset.xlsx")  # Replace 'dataset.xlsx' with your file name

# 2. Split Features and Target
X = data.drop(columns=["flow_rate", "year", "station"])
Y = data["flow_rate"]

# 3. Define Numerical and Categorical Features
numerical_features = ["precipitation", "temperature", "slope", "basin_area"]
categorical_features = ["land_use", "aspect", "awc_top", "oc_top"]

# 4. Adjust Data Types of Categorical Features
X[categorical_features] = X[categorical_features].astype(str)

# 5. Split Data into Train and Test Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 6. Prepare Data for CatBoost
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

# 8. Predictions and Performance Metrics
Y_pred_cat = cat_model.predict(test_pool)

cat_r2 = r2_score(Y_test, Y_pred_cat)
cat_rmse = mean_squared_error(Y_test, Y_pred_cat, squared=False)
cat_mae = mean_absolute_error(Y_test, Y_pred_cat)

print(f"CatBoost Test Set R²: {cat_r2:.4f}")
print(f"CatBoost Test Set RMSE: {cat_rmse:.4f}")
print(f"CatBoost Test Set MAE: {cat_mae:.4f}")

# 9. Feature Importance
feature_importance = pd.DataFrame({
    "Feature": cat_model.feature_names_,
    "Importance": cat_model.get_feature_importance(train_pool)
}).sort_values(by="Importance", ascending=False)

# Feature Importance Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance["Feature"], feature_importance["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("CatBoost Feature Importance")
plt.gca().invert_yaxis()
plt.show()

# 10. Visualization
# Actual vs Predicted Values
plt.figure(figsize=(12, 6))
plt.scatter(Y_test, Y_pred_cat, alpha=0.7, label="Estimated", marker='o', edgecolor="k", s=90)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "r--", label="Reference Line")
plt.title("Comparison of Actual and Estimated Values (CatBoost)")
plt.xlabel("Actual Values")
plt.ylabel("Estimated Values")
plt.legend()
plt.show()

# Residual Analysis
residuals = Y_test - Y_pred_cat
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=20, edgecolor="k", alpha=0.7)
plt.title("Error Distribution (Residual Analysis)")
plt.xlabel("Error (Actual - Estimated)")
plt.ylabel("Frequency")
plt.show()

# 11. Save Metrics and Results
results = pd.DataFrame({
    "Metric": ["R²", "RMSE", "MAE"],
    "Value": [cat_r2, cat_rmse, cat_mae]
})

results.to_excel("catboost_error_results.xlsx", index=False)
feature_importance.to_excel("catboost_feature_importance.xlsx", index=False)

print("\nPerformance metrics saved to 'catboost_error_results.xlsx'.")
print("Feature importances saved to 'catboost_feature_importance.xlsx'.")

# Function to Add Noise for Data Augmentation
def add_noise(data, noise_level=0.1, repetitions=100, seed=42):
    """
    Adds noise to numerical and categorical data for data augmentation.
    """
    np.random.seed(seed)
    augmented_data = []

    for _ in range(repetitions):
        noisy_data = data.copy()

        # Add noise to numerical features
        for col in numerical_features:
            noise = np.random.normal(0, noise_level * noisy_data[col].std(), size=len(noisy_data))
            noisy_data[col] += noise

        # Add random changes to categorical features
        for col in categorical_features:
            num_changes = int(noise_level * len(noisy_data))
            random_indices = np.random.choice(noisy_data.index, size=num_changes, replace=False)
            unique_values = noisy_data[col].unique()
            noisy_data.loc[random_indices, col] = np.random.choice(unique_values, size=num_changes)

        augmented_data.append(noisy_data)

    # Combine augmented datasets
    return pd.concat(augmented_data, ignore_index=True)

# Create Augmented Dataset (10 repetitions, 10% noise)
augmented_data = add_noise(X, noise_level=0.1, repetitions=10)

# Combine Augmented Data with Original Data
augmented_X = pd.concat([X, augmented_data], ignore_index=True)
augmented_Y = pd.concat([Y] * (1 + 10), ignore_index=True)

# Print Dataset Sizes
print(f"Original dataset size: {X.shape[0]}")
print(f"Augmented dataset size: {augmented_X.shape[0]}")

# Split Augmented Data into Train and Test Sets
X_train_aug, X_test_aug, Y_train_aug, Y_test_aug = train_test_split(
    augmented_X, augmented_Y, test_size=0.2, random_state=42
)

# Prepare Augmented Data for CatBoost
train_pool_aug = Pool(X_train_aug, Y_train_aug, cat_features=categorical_features)
test_pool_aug = Pool(X_test_aug, Y_test_aug, cat_features=categorical_features)

# Train CatBoost Model (Augmented Data)
print("\nTraining CatBoost Model (Augmented Data)...")
cat_model_aug = CatBoostRegressor(
    iterations=1000,
    depth=10,
    learning_rate=0.03,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=100
)
cat_model_aug.fit(train_pool_aug)

# Performance Metrics for Augmented Data
Y_pred_cat_aug = cat_model_aug.predict(test_pool_aug)
cat_r2_aug = r2_score(Y_test_aug, Y_pred_cat_aug)
cat_rmse_aug = mean_squared_error(Y_test_aug, Y_pred_cat_aug, squared=False)
cat_mae_aug = mean_absolute_error(Y_test_aug, Y_pred_cat_aug)

print(f"\nCatBoost (Augmented Data) Test Set R²: {cat_r2_aug:.4f}")
print(f"CatBoost (Augmented Data) Test Set RMSE: {cat_rmse_aug:.4f}")
print(f"CatBoost (Augmented Data) Test Set MAE: {cat_mae_aug:.4f}")

# Visualization
plt.figure(figsize=(12, 6))
plt.scatter(Y_test_aug, Y_pred_cat_aug, alpha=0.7, label="Estimated (Augmented)", marker='o', edgecolor="k", s=90)
plt.plot([Y_test_aug.min(), Y_test_aug.max()], [Y_test_aug.min(), Y_test_aug.max()], "r--", label="Reference Line")
plt.title("Comparison of Actual and Estimated Values (Augmented Data)")
plt.xlabel("Actual Values")
plt.ylabel("Estimated Values")
plt.legend()
plt.show()

# Residual Analysis: Augmented vs Original
residuals_original = Y_test - Y_pred_cat
residuals_augmented = Y_test_aug - Y_pred_cat_aug

plt.figure(figsize=(12, 6))
plt.hist(residuals_original, bins=20, edgecolor="k", alpha=0.6, label="Residuals (Original)")
plt.hist(residuals_augmented, bins=20, edgecolor="k", alpha=0.6, label="Residuals (Augmented)")
plt.title("Error Distribution (Residual Analysis)")
plt.xlabel("Error (Actual - Estimated)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Save Model
cat_model.save_model("catboost_model.cbm")
print("Model saved as 'catboost_model.cbm'.")

