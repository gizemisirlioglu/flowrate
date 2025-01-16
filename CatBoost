import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool

# 1. Load Dataset
data = pd.read_excel("dataset.xlsx")  # Replace 'dataset.xlsx' with your file name

# 2. Split Features and Target
X = data.drop(columns=["flow_rate", "year", "station"])
Y = data["flow_rate"]

# 3. Identify Numerical and Categorical Features
numerical_features = ["precipitation", "temperature", "slope", "basin_area"]
categorical_features = ["land_use", "aspect", "awc_top", "oc_top"]

# 4. Fix Categorical Variable Types
X[categorical_features] = X[categorical_features].astype(str)

# 5. Split Data into Training and Testing Sets
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

# 8. Predictions and Performance
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

# 11. Save Results
results = pd.DataFrame({
    "Metric": ["R²", "RMSE", "MAE"],
    "Value": [cat_r2, cat_rmse, cat_mae]
})

results.to_excel("catboost_error_results.xlsx", index=False)
feature_importance.to_excel("catboost_feature_importance.xlsx", index=False)

# Save Predictions
predictions_df = pd.DataFrame({
    "Actual": Y_test.values,
    "Predicted": Y_pred_cat
})
predictions_df.to_excel("catboost_predictions.xlsx", index=False)

print("\nPerformance metrics saved to 'catboost_error_results.xlsx'.")
print("Feature importance saved to 'catboost_feature_importance.xlsx'.")
print("Predicted and actual values saved to 'catboost_predictions.xlsx'.")

