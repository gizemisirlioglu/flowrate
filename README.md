# Flow rate Estimation

This project was developed to predict flow rate data and visualize the results.

Contents
1. About Project
2. Used Code Files
3. Requirements
4. How to Use
5. Data Format


1. About Project 
This project was written to analyze and report flow rates from the GRDC station with open source data such as topography, climate, land use and soil properties.
Objectives:
- Cleaning of open source data
- Testing of all models
- Estimating flow rate with final model
- Testing with noise data
- Estimating flow rate for the next 100 years



2. Used Code Files
- **data_cleaning.py**: Ham verilerin temizlenmesi ve işlenmesi.
- **flowrate_calculation.py**: Akış hızlarının hesaplanması için temel algoritmalar.
- **visualization.py**: Akış hızlarını grafiksel olarak görselleştirmek.

---

Requirements
To run the project, the following software and libraries are required:

Software
- Python 3.8+
- GDAL (required for raster and DEM operations)
Python Libraries
- NumPy: For numerical operations
- Pandas: For data manipulation and analysis
- Matplotlib: For data visualization
- Rasterio: For raster data processing
- scikit-learn: For machine learning tasks
- openpyxl: For handling Excel files

To install the required Python libraries, use the following command in the terminal:
pip install -r requirements

