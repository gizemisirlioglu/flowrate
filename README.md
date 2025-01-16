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
- **missingdata.py**: Estimation of missing data at stations.
- **dem_nodata.py**: Filling in missing areas in DEM data.
- **testingallmodels.py**: All algorithms for flow rate estimation.
- **CatBoost.py**: The most reliable method.
- **noisedata.py**: Testing the method with noise data.
- **next100year.py**: Predicting the flow rate for the next 100 years.

---

3.Requirements
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

4. How to Use

All codes can be tested with the sample data sets provided on GitHub by uploading the relevant file in the code instead of uploading.

5. Data Format

- Codes for flow amount determinations are prepared according to the .xlsx file type, it can be converted to .csv format with the necessary correction.
- If analysis is required for the DEM image, the file type is set to .tif.
