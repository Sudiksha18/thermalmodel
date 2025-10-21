#!/usr/bin/env python3
"""
MAT file loader for thermal anomaly detection system.
"""

import numpy as np
import scipy.io as sio
import torch
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

def load_mat_file(file_path: Union[str, Path], 
                  normalize: bool = True,
                  target_size: Optional[Tuple[int, int]] = None,
                  temperature_range: Optional[Tuple[float, float]] = None) -> Dict:
    """
    Load thermal data from .mat file.
    
    Args:
        file_path: Path to .mat file
        normalize: Whether to normalize the data
        target_size: Target size for resizing (height, width)
        temperature_range: Temperature range for normalization
        
    Returns:
        Dictionary containing thermal data and metadata
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"MAT file not found: {file_path}")
    
    logger.info(f"Loading thermal data from {file_path}")
    
    try:
        # Load .mat file
        mat_data = sio.loadmat(file_path)
        
        # Extract thermal data
        thermal_data = mat_data['thermal_data']
        
        # Handle NaN values
        if np.isnan(thermal_data).any():
            nan_count = np.sum(np.isnan(thermal_data))
            logger.warning(f"Found {nan_count} NaN values in thermal data ({nan_count/thermal_data.size*100:.2f}%), replacing with mean")
            thermal_data = np.nan_to_num(thermal_data, nan=np.nanmean(thermal_data))
            logger.info(f"After NaN replacement - Valid pixels: {thermal_data.size}")
        
        # Convert to float32
        thermal_data = thermal_data.astype(np.float32)
        
        logger.info(f"Original thermal data shape: {thermal_data.shape}")
        logger.info(f"Temperature range: {thermal_data.min():.2f}°C to {thermal_data.max():.2f}°C")
        logger.info(f"Data statistics: mean={thermal_data.mean():.2f}°C, std={thermal_data.std():.2f}°C")
        
        # Resize if target size is specified
        if target_size is not None:
            original_shape = thermal_data.shape
            h, w = target_size
            thermal_data = cv2.resize(thermal_data, (w, h), interpolation=cv2.INTER_LINEAR)
            logger.info(f"Resized from {original_shape} to {thermal_data.shape}")
            logger.info(f"Resolution change: {original_shape[0]*original_shape[1]} → {h*w} pixels ({(h*w)/(original_shape[0]*original_shape[1])*100:.2f}% of original)")
        
        # Normalize if requested
        if normalize:
            if temperature_range is not None:
                min_temp, max_temp = temperature_range
                thermal_data = np.clip(thermal_data, min_temp, max_temp)
                thermal_data = (thermal_data - min_temp) / (max_temp - min_temp)
                logger.info(f"Normalized to range [0, 1] using temp range [{min_temp}, {max_temp}]")
            else:
                # Use data min/max for normalization
                min_val, max_val = thermal_data.min(), thermal_data.max()
                if max_val > min_val:
                    thermal_data = (thermal_data - min_val) / (max_val - min_val)
                    logger.info(f"Normalized to range [0, 1] using data range [{min_val:.2f}, {max_val:.2f}]")
        
        # Extract anomaly mask if available
        anomaly_mask = None
        if 'anomaly_mask' in mat_data:
            anomaly_mask = mat_data['anomaly_mask'].astype(np.uint8)
            if target_size is not None:
                h, w = target_size
                anomaly_mask = cv2.resize(anomaly_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            logger.info(f"Loaded anomaly mask with {np.sum(anomaly_mask)} positive pixels")
        
        # Prepare return data
        result = {
            'thermal_data': thermal_data,
            'anomaly_mask': anomaly_mask,
            'has_annotations': anomaly_mask is not None,
            'metadata': {
                'filename': file_path.name,
                'original_shape': mat_data.get('shape', thermal_data.shape),
                'crs': mat_data.get('crs', ''),
                'transform': mat_data.get('transform', None),
                'source_format': 'MAT'
            }
        }
        
        logger.info(f"Successfully loaded thermal data from {file_path}")
        return result
        
    except Exception as e:
        logger.error(f"Error loading MAT file {file_path}: {e}")
        raise

def save_mat_file(thermal_data: np.ndarray,
                  file_path: Union[str, Path],
                  anomaly_mask: Optional[np.ndarray] = None,
                  metadata: Optional[Dict] = None) -> None:
    """
    Save thermal data to .mat file.
    
    Args:
        thermal_data: Thermal data array
        file_path: Output file path
        anomaly_mask: Optional anomaly mask
        metadata: Optional metadata dictionary
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    mat_data = {
        'thermal_data': thermal_data,
        'shape': thermal_data.shape
    }
    
    if anomaly_mask is not None:
        mat_data['anomaly_mask'] = anomaly_mask
        mat_data['has_annotations'] = True
    else:
        mat_data['has_annotations'] = False
    
    if metadata is not None:
        mat_data.update(metadata)
    
    sio.savemat(file_path, mat_data)
    logger.info(f"Saved thermal data to {file_path}")