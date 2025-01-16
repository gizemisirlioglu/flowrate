import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

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

# 6. Data Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ============================= Model Evaluation Function ================================
def evaluate_model_with_cross_val(model, X_train, X_test, Y_train, Y_test, model_name):
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # Performance Metrics
    r2 = r2_score(Y_test, Y_pred)
    rmse = mean_squared_error(Y_test, Y_pred, squared=False)
    mae = mean_absolute_error(Y_test, Y_pred)

    # Cross-Validation
    cross_val_r2 = cross_val_score(model, X_train, Y_train, cv=5, scoring='r2').mean()
    cross_val_rmse = np.mean(
        -cross_val_score(model, X_train, Y_train, cv=5, scoring='neg_root_mean_squared_error')
    )

    print(f"{model_name} Test Set R²: {r2:.4f}")
    print(f"{model_name} Test Set RMSE: {rmse:.4f}")
    print(f"{model_name} Test Set MAE: {mae:.4f}")
    print(f"{model_name} Cross-Validation R²: {cross_val_r2:.4f}")
    print(f"{model_name} Cross-Validation RMSE: {cross_val_rmse:.4f}")

    return r2, rmse, mae, cross_val_r2, cross_val_rmse, Y_pred

# ============================= Model Evaluation ================================

# XGBoost
print("\nTraining XGBoost Model...")
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=700,
    learning_rate=0.05,
    max_depth=10,
    subsample=0.8,
    random_state=42
)
xgb_r2, xgb_rmse, xgb_mae, xgb_cv_r2, xgb_cv_rmse, Y_pred_xgb = evaluate_model_with_cross_val(
    xgb_model, X_train_processed, X_test_processed, Y_train, Y_test, "XGBoost"
)

# LightGBM
print("\nTraining LightGBM Model...")
lgb_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=700,
    learning_rate=0.05,
    max_depth=10,
    random_state=42
)
lgb_r2, lgb_rmse, lgb_mae, lgb_cv_r2, lgb_cv_rmse, Y_pred_lgb = evaluate_model_with_cross_val(
    lgb_model, X_train_processed, X_test_processed, Y_train, Y_test, "LightGBM"
)

# CatBoost
print("\nTraining CatBoost Model...")
cat_model = CatBoostRegressor(
    cat_features=[X.columns.get_loc(c) for c in categorical_features],
    verbose=0,
    random_seed=42
)
cat_r2, cat_rmse, cat_mae, cat_cv_r2, cat_cv_rmse, Y_pred_cat = evaluate_model_with_cross_val(
    cat_model, X_train, X_test, Y_train, Y_test, "CatBoost"
)

# Random Forest
print("\nTraining Random Forest Model...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_r2, rf_rmse, rf_mae, rf_cv_r2, rf_cv_rmse, Y_pred_rf = evaluate_model_with_cross_val(
    rf_model, X_train_processed, X_test_processed, Y_train, Y_test, "Random Forest"
)

# ============================ Save Performance Metrics ================================
results = {
    "Model": ["XGBoost", "LightGBM", "CatBoost", "Random Forest"],
    "R²": [xgb_r2, lgb_r2, cat_r2, rf_r2],
    "RMSE": [xgb_rmse, lgb_rmse, cat_rmse, rf_rmse],
    "MAE": [xgb_mae, lgb_mae, cat_mae, rf_mae],
    "Cross-Validation R²": [xgb_cv_r2, lgb_cv_r2, cat_cv_r2, rf_cv_r2],
    "Cross-Validation RMSE": [xgb_cv_rmse, lgb_cv_rmse, cat_cv_rmse, rf_cv_rmse]
}
results_df = pd.DataFrame(results)
results_df.to_excel("model_performance.xlsx", index=False)
print("\nPerformance metrics saved to 'model_performance.xlsx'.")

# ============================= Visualization ================================
plt.figure(figsize=(12, 6))

# XGBoost
plt.scatter(Y_test, Y_pred_xgb, alpha=0.7, label="XGBoost", marker='o', edgecolor="k", s=90)

# LightGBM
plt.scatter(Y_test, Y_pred_lgb, alpha=0.7, label="LightGBM", marker='s', edgecolor="k", s=90)

# CatBoost
plt.scatter(Y_test, Y_pred_cat, alpha=0.7, label="CatBoost", marker='x', edgecolor="k", s=90)

# Random Forest
plt.scatter(Y_test, Y_pred_rf, alpha=0.7, label="Random Forest", marker='*', edgecolor="k", s=90)

# Reference Line
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "r--", label="Reference Line")

# Labels and Title
plt.title("Comparison of Actual and Estimated Values")
plt.xlabel("Actual Values")
plt.ylabel("Estimated Values")
plt.legend()
plt.show()

