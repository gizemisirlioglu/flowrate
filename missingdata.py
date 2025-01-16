import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Folder containing Excel files
input_folder = "input_folder_path_here"  # Replace with the folder containing your input Excel files
output_folder = "output_folder_path_here"  # Replace with the folder where you want to save the results

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate over all Excel files in the folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.xlsx'):
        input_file = os.path.join(input_folder, file_name)

        # Load data
        df = pd.read_excel(input_file)

        # Check and clean column names
        df.columns = [col.strip() for col in df.columns]
        df.rename(columns={'Year': 'Year', 'Month': 'Month', 'Day': 'Day', 'Value': 'Value'}, inplace=True)

        # Check for missing values
        print(f"Processing file: {file_name}")
        print("Missing value check:")
        print(df[['Year', 'Month', 'Day']].isnull().sum())

        # Create a date column
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

        # Generate a full date range
        full_date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
        df_full = pd.DataFrame({'Date': full_date_range})
        df_full = df_full.merge(df, on='Date', how='left')

        # Add new features
        df_full['Julian_Date'] = df_full['Date'].apply(lambda x: x.to_julian_date())
        df_full['Day_of_Year'] = df_full['Date'].dt.dayofyear
        df_full['Month'] = df_full['Date'].dt.month
        df_full['Weekday'] = df_full['Date'].dt.weekday
        df_full['Is_Weekend'] = df_full['Weekday'].apply(lambda x: 1 if x >= 5 else 0)

        # Prepare data for RandomForest
        X = df_full[['Julian_Date', 'Day_of_Year', 'Month', 'Weekday', 'Is_Weekend']]
        y = df_full['Value']

        # Separate known and unknown values
        known_data = df_full[df_full['Value'].notnull()]
        unknown_data = df_full[df_full['Value'].isnull()]

        if known_data.empty or unknown_data.empty:
            print(f"Skipping file {file_name} due to insufficient data.")
            continue

        X_known = known_data[['Julian_Date', 'Day_of_Year', 'Month', 'Weekday', 'Is_Weekend']]
        y_known = known_data['Value']
        X_unknown = unknown_data[['Julian_Date', 'Day_of_Year', 'Month', 'Weekday', 'Is_Weekend']]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, test_size=0.2, random_state=42)

        if X_train.empty or X_test.empty:
            print(f"Skipping file {file_name} due to insufficient training/test data.")
            continue

        # RandomForest Model
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)

        # Predict missing values
        unknown_data['Value'] = rf_model.predict(X_unknown)

        # Combine known and predicted values
        df_filled = pd.concat([known_data, unknown_data]).sort_values(by='Date')

        # Save filled data
        filled_data_output = os.path.join(output_folder, f"{file_name.split('.')[0]}_filled.xlsx")
        df_filled.to_excel(filled_data_output, index=False)
        print(f"Filled data saved for {file_name}: {filled_data_output}")

        # Performance Metrics
        mae = mean_absolute_error(y_test, rf_preds)
        rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
        r2 = r2_score(y_test, rf_preds)
        print(f"RandomForest Performance for {file_name}:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2: {r2:.2f}\n")

        # Save predictions and performance
        predictions_output = os.path.join(output_folder, f"{file_name.split('.')[0]}_predictions.xlsx")
        pd.DataFrame({
            'Date': X_test.index,
            'Actual': y_test,
            'Predicted': rf_preds
        }).to_excel(predictions_output, index=False)
        print(f"Predictions saved for {file_name}: {predictions_output}")

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.scatter(df_filled['Date'], df_filled['Value'], label='All Values', color='blue', s=10)
        plt.scatter(unknown_data['Date'], unknown_data['Value'], label='Predicted Values', color='red', s=20)
        plt.legend()
        plt.title(f"Estimated Values: {file_name}", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.grid(True)
        plot_output = os.path.join(output_folder, f"{file_name.split('.')[0]}_visualization.png")
        plt.savefig(plot_output)
        plt.show()
        print(f"Visualization saved for {file_name}: {plot_output}")
