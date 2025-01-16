from osgeo import gdal
import os

def fill_dem_nodata(input_path, output_path, max_distance=50, smoothing_iterations=0):
    """
    Fills gaps in a Digital Elevation Model (DEM) using GDAL's FillNodata function.

    Args:
        input_path (str): Path to the input DEM file.
        output_path (str): Path to save the filled DEM file.
        max_distance (int): Maximum search distance (in pixels).
        smoothing_iterations (int): Number of smoothing iterations.
    """
    # Open the input DEM file
    src_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    if not src_ds:
        print("The DEM file could not be opened.")
        return

    # Get the raster band
    band = src_ds.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    print(f"NoData value: {nodata_value}")

    # Create a temporary raster
    driver = gdal.GetDriverByName('GTiff')
    temp_ds = driver.CreateCopy('temp_filled.tif', src_ds, 0)
    temp_band = temp_ds.GetRasterBand(1)

    # Define the NoData value if not already set (e.g., -9999)
    if nodata_value is None:
        nodata_value = -9999
        temp_band.SetNoDataValue(nodata_value)

    # Apply the fill operation
    gdal.FillNodata(targetBand=temp_band,
                    maskBand=None,
                    maxSearchDist=max_distance,
                    smoothingIterations=smoothing_iterations)

    # Save the output file
    temp_band.FlushCache()
    temp_ds.FlushCache()
    temp_ds = None  # Release from memory

    # Move the temporary file to the target location
    os.rename('temp_filled.tif', output_path)
    print(f"Gaps filled, and the new file is saved as '{output_path}'.")

# Define input and output file paths
input_dem = r"input_dem_path_here.tif"  # Replace with the input DEM file path
output_dem = r"output_dem_path_here.tif"  # Replace with the desired output DEM file path

# Start the fill operation
fill_dem_nodata(input_dem, output_dem, max_distance=50)
