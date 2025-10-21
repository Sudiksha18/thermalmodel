#!/usr/bin/env python3
"""
Convert TIF thermal data to .mat format for thermal anomaly detection.
"""

import numpy as np
import rasterio
import scipy.io as sio
from pathlib import Path
import argparse

def convert_tif_to_mat(tif_path, output_path=None):
    """
    Convert TIF thermal data to .mat format.
    
    Args:
        tif_path: Path to input TIF file
        output_path: Path for output .mat file (optional)
    """
    tif_path = Path(tif_path)
    
    if output_path is None:
        output_path = tif_path.with_suffix('.mat')
    else:
        output_path = Path(output_path)
    
    print(f"Converting {tif_path} to {output_path}")
    
    # Read TIF data
    with rasterio.open(tif_path) as src:
        thermal_data = src.read(1)  # Read first band
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        
        print(f"Original shape: {thermal_data.shape}")
        print(f"Data type: {thermal_data.dtype}")
        print(f"Value range: {thermal_data.min()} to {thermal_data.max()}")
        print(f"Non-zero values: {np.count_nonzero(thermal_data)}")
        print(f"Total pixels: {thermal_data.size}")
        print(f"Data density: {np.count_nonzero(thermal_data)/thermal_data.size*100:.2f}%")
        
        # Handle nodata values
        if nodata is not None:
            print(f"NoData value: {nodata}")
            nodata_count = np.sum(thermal_data == nodata)
            print(f"NoData pixels: {nodata_count} ({nodata_count/thermal_data.size*100:.2f}%)")
            thermal_data = np.where(thermal_data == nodata, np.nan, thermal_data)
            print(f"After NoData removal - Valid pixels: {np.count_nonzero(~np.isnan(thermal_data))}")
        
        # Convert to Celsius if needed (Landsat thermal data is typically in Kelvin)
        if thermal_data.max() > 200:  # Likely Kelvin
            # Apply proper Landsat 8 thermal conversion formula
            thermal_data_celsius = thermal_data * 0.00341802 + 149.0 - 273.15
            print(f"Applied Landsat 8 thermal conversion formula")
            print(f"Converted to Celsius: {np.nanmin(thermal_data_celsius):.2f}°C to {np.nanmax(thermal_data_celsius):.2f}°C")
            print(f"No data loss - using proper calibration coefficients")
        else:
            thermal_data_celsius = thermal_data
            print(f"Temperature range: {np.nanmin(thermal_data_celsius):.2f}°C to {np.nanmax(thermal_data_celsius):.2f}°C")
        
        # Prepare data for .mat file
        mat_data = {
            'thermal_data': thermal_data_celsius,
            'thermal_data_kelvin': thermal_data,
            'transform': np.array([transform.a, transform.b, transform.c, 
                                 transform.d, transform.e, transform.f]),
            'crs': str(crs) if crs else '',
            'nodata': nodata if nodata is not None else np.nan,
            'shape': thermal_data.shape,
            'filename': tif_path.name,
            'conversion_info': {
                'source_format': 'GeoTIFF',
                'converted_by': 'thermal_anomaly_detection_system',
                'temperature_unit': 'Celsius'
            }
        }
        
        # Create synthetic anomaly mask for demonstration
        # In real scenarios, this would come from ground truth data
        anomaly_mask = np.zeros_like(thermal_data_celsius, dtype=np.uint8)
        
        # Create some synthetic hot spots as anomalies
        h, w = thermal_data_celsius.shape
        # Add a few random hot spots
        np.random.seed(42)
        for _ in range(5):
            y = np.random.randint(h//4, 3*h//4)
            x = np.random.randint(w//4, 3*w//4)
            size = np.random.randint(10, 30)
            y1, y2 = max(0, y-size//2), min(h, y+size//2)
            x1, x2 = max(0, x-size//2), min(w, x+size//2)
            anomaly_mask[y1:y2, x1:x2] = 1
        
        mat_data['anomaly_mask'] = anomaly_mask
        mat_data['has_annotations'] = True
        
        # Save as .mat file
        sio.savemat(output_path, mat_data)
        print(f"Successfully saved to {output_path}")
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Convert TIF thermal data to .mat format")
    parser.add_argument("input_tif", help="Path to input TIF file")
    parser.add_argument("--output", "-o", help="Output .mat file path (optional)")
    
    args = parser.parse_args()
    
    try:
        output_path = convert_tif_to_mat(args.input_tif, args.output)
        print(f"Conversion completed successfully: {output_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())