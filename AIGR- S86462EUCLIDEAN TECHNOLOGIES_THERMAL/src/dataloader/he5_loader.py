#!/usr/bin/env python3
"""
HE5 Thermal Data Loader for Euclidean Technologies Thermal Anomaly Detection.
Handles reading and processing of .he5 thermal files with minimal temperature data loss.
"""

import os
import h5py
import numpy as np
import torch
import rasterio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

# Import MAT loader
from .mat_loader import load_mat_file

logger = logging.getLogger(__name__)


class HE5ThermalLoader:
    """
    Loader for HE5 thermal datasets (Landsat-8/9, FLAME, KAIST, MIRSat-QL).
    Preserves temperature data with minimal preprocessing.
    """
    
    def __init__(self, 
                 normalize: bool = True,
                 target_size: Optional[Tuple[int, int]] = None,
                 temperature_range: Optional[Tuple[float, float]] = None):
        """
        Initialize HE5 thermal loader.
        
        Args:
            normalize: Whether to normalize temperature values
            target_size: Target size for resizing (height, width)
            temperature_range: Valid temperature range (min_temp, max_temp) in Kelvin
        """
        self.normalize = normalize
        self.target_size = target_size
        self.temperature_range = temperature_range or (250.0, 400.0)  # Typical thermal range
        
        # Common thermal band names in different datasets
        self.thermal_band_names = [
            'ST_B10',  # Landsat-8/9 thermal infrared
            'thermal',
            'band10',
            'TIR',
            'LWIR',
            'temperature',
            'Temperature',
            'LST'  # Land Surface Temperature
        ]
    
    def load_he5_file(self, file_path: Union[str, Path]) -> Dict:
        """
        Load thermal data from HE5 file.
        
        Args:
            file_path: Path to HE5 file
            
        Returns:
            Dictionary containing thermal data and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"HE5 file not found: {file_path}")
        
        logger.info(f"Loading HE5 file: {file_path}")
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Find thermal band
                thermal_data, band_info = self._find_thermal_band(f)
                
                if thermal_data is None:
                    raise ValueError(f"No thermal band found in {file_path}")
                
                # Extract metadata
                metadata = self._extract_metadata(f, band_info)
                
                # Process thermal data
                processed_data = self._process_thermal_data(thermal_data)
                
                return {
                    'thermal_data': processed_data,
                    'metadata': metadata,
                    'file_path': str(file_path),
                    'band_name': band_info['name'],
                    'original_shape': thermal_data.shape
                }
                
        except Exception as e:
            logger.error(f"Error loading HE5 file {file_path}: {str(e)}")
            raise
    
    def load_tif_file(self, file_path: Union[str, Path]) -> Dict:
        """
        Load thermal data from TIF file (e.g., Landsat thermal bands).
        
        Args:
            file_path: Path to TIF file
            
        Returns:
            Dictionary containing thermal data and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"TIF file not found: {file_path}")
        
        logger.info(f"Loading TIF file: {file_path}")
        
        try:
            with rasterio.open(file_path) as src:
                # Read thermal data
                thermal_data = src.read(1).astype(np.float32)
                
                # Get metadata
                metadata = {
                    'crs': src.crs,
                    'transform': src.transform,
                    'width': src.width,
                    'height': src.height,
                    'dtype': str(thermal_data.dtype),
                    'nodata': src.nodata
                }
                
                # Process thermal data
                processed_data = self._process_thermal_data(thermal_data)
                
                return {
                    'thermal_data': processed_data,
                    'metadata': metadata,
                    'file_path': str(file_path),
                    'band_name': 'thermal',
                    'original_shape': thermal_data.shape
                }
                
        except Exception as e:
            logger.error(f"Error loading TIF file {file_path}: {str(e)}")
            raise
    
    def _find_thermal_band(self, h5_file) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Find thermal band in HE5 file."""
        
        def search_groups(group, path=""):
            """Recursively search for thermal bands."""
            for key in group.keys():
                current_path = f"{path}/{key}" if path else key
                
                # Check if this is a thermal band
                if any(thermal_name.lower() in key.lower() for thermal_name in self.thermal_band_names):
                    if isinstance(group[key], h5py.Dataset):
                        return group[key][:], {'name': key, 'path': current_path}
                
                # Recurse into groups
                if isinstance(group[key], h5py.Group):
                    result = search_groups(group[key], current_path)
                    if result[0] is not None:
                        return result
            
            return None, None
        
        return search_groups(h5_file)
    
    def _extract_metadata(self, h5_file, band_info: Dict) -> Dict:
        """Extract metadata from HE5 file."""
        metadata = {
            'band_info': band_info,
            'attributes': {}
        }
        
        # Extract global attributes
        for attr_name in h5_file.attrs.keys():
            try:
                metadata['attributes'][attr_name] = h5_file.attrs[attr_name]
            except:
                pass
        
        # Extract geospatial information if available
        try:
            # Look for common geospatial datasets
            geospatial_keys = ['Latitude', 'Longitude', 'lat', 'lon', 'x', 'y']
            for key in geospatial_keys:
                if key in h5_file:
                    metadata[key.lower()] = h5_file[key][:]
        except:
            pass
        
        return metadata
    
    def _process_thermal_data(self, thermal_data: np.ndarray) -> torch.Tensor:
        """
        Process thermal data with minimal temperature loss.
        
        Args:
            thermal_data: Raw thermal data array
            
        Returns:
            Processed thermal tensor
        """
        # Convert to float32 to preserve precision
        data = thermal_data.astype(np.float32)
        
        # Handle invalid/missing values
        if hasattr(thermal_data, 'mask'):
            # Masked array
            data = np.ma.filled(thermal_data, np.nan)
        
        # Replace invalid values with interpolation
        if np.isnan(data).any():
            data = self._interpolate_missing_values(data)
        
        # Clip to valid temperature range
        data = np.clip(data, self.temperature_range[0], self.temperature_range[1])
        
        # Normalize if requested
        if self.normalize:
            data = self._normalize_temperature(data)
        
        # Resize if requested
        if self.target_size:
            data = self._resize_data(data, self.target_size)
        
        # Convert to tensor
        tensor = torch.from_numpy(data).float()
        
        # Add channel dimension if needed
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def _interpolate_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Interpolate missing values in thermal data."""
        from scipy.interpolate import griddata
        
        # Find valid points
        valid_mask = ~np.isnan(data)
        if not valid_mask.any():
            logger.warning("No valid thermal data found, using mean value")
            return np.full_like(data, np.nanmean(self.temperature_range))
        
        if valid_mask.all():
            return data
        
        # Get coordinates
        y_coords, x_coords = np.mgrid[0:data.shape[0], 0:data.shape[1]]
        
        # Valid points
        valid_points = np.column_stack((y_coords[valid_mask], x_coords[valid_mask]))
        valid_values = data[valid_mask]
        
        # Invalid points to interpolate
        invalid_points = np.column_stack((y_coords[~valid_mask], x_coords[~valid_mask]))
        
        # Interpolate
        try:
            interpolated_values = griddata(
                valid_points, 
                valid_values, 
                invalid_points, 
                method='linear',
                fill_value=np.mean(valid_values)
            )
            
            data[~valid_mask] = interpolated_values
        except Exception as e:
            logger.warning(f"Interpolation failed: {e}, using mean value")
            data[~valid_mask] = np.mean(valid_values)
        
        return data
    
    def _normalize_temperature(self, data: np.ndarray) -> np.ndarray:
        """Normalize temperature data to [0, 1] range."""
        min_temp, max_temp = self.temperature_range
        normalized = (data - min_temp) / (max_temp - min_temp)
        return np.clip(normalized, 0.0, 1.0)
    
    def _resize_data(self, data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize thermal data using bilinear interpolation."""
        import cv2
        
        # Use OpenCV for high-quality resizing
        resized = cv2.resize(
            data, 
            (target_size[1], target_size[0]),  # OpenCV uses (width, height)
            interpolation=cv2.INTER_LINEAR
        )
        
        return resized
    
    def get_thermal_stats(self, thermal_data: Union[np.ndarray, torch.Tensor]) -> Dict:
        """Get statistics of thermal data."""
        if isinstance(thermal_data, torch.Tensor):
            data = thermal_data.numpy()
        else:
            data = thermal_data
        
        return {
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'shape': data.shape,
            'dtype': str(data.dtype)
        }


def load_thermal_file(file_path: Union[str, Path], **kwargs) -> Dict:
    """
    Convenience function to load thermal files (HE5, TIF, or MAT).
    
    Args:
        file_path: Path to thermal file
        **kwargs: Additional arguments for HE5ThermalLoader
        
    Returns:
        Dictionary containing thermal data and metadata
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    if extension == '.he5':
        loader = HE5ThermalLoader(**kwargs)
        return loader.load_he5_file(file_path)
    elif extension in ['.tif', '.tiff']:
        loader = HE5ThermalLoader(**kwargs)
        return loader.load_tif_file(file_path)
    elif extension == '.mat':
        return load_mat_file(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {extension}")


if __name__ == "__main__":
    # Test loading
    loader = HE5ThermalLoader(normalize=True, target_size=(512, 512))
    
    # Example usage
    test_file = "data/LC08_L2SP_138045_20250215_20250226_02_T1_ST_B10.TIF"
    if os.path.exists(test_file):
        result = loader.load_tif_file(test_file)
        print(f"Loaded thermal data: {result['thermal_data'].shape}")
        print(f"Stats: {loader.get_thermal_stats(result['thermal_data'])}")
    else:
        print(f"Test file not found: {test_file}")