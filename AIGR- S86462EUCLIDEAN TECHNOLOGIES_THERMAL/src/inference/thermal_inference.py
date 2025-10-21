#!/usr/bin/env python3
"""
Inference module for Thermal Anomaly Detection System.
"""

import os
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union

from src.models.swin_unet import create_swin_unet
from src.dataloader.he5_loader import load_thermal_file
from src.inference.export_submission import SubmissionGenerator

logger = logging.getLogger(__name__)

class ThermalAnomalyInference:
    """Class for performing inference with trained thermal anomaly detection models."""
    
    def __init__(self, config: dict, model_path: Optional[str] = None):
        """
        Initialize the inference system.
        
        Args:
            config: Configuration dictionary
            model_path: Path to trained model weights (optional)
        """
        self.config = config
        self.device = torch.device(config['hardware'].get('device', 'cuda'))
        
        # Create model
        self.model = create_swin_unet(config['model'])
        self.model = self.model.to(self.device)
        
        if model_path:
            self._load_model(model_path)
            
        # Set to evaluation mode
        self.model.eval()
        
        # Initialize submission generator
        self.submission_gen = SubmissionGenerator(
            startup_name=config['inference'].get('startup_name', 'EuclideanTechnologies'),
            output_dir=config['inference'].get('output_dir', 'submission/')
        )
        
        logger.info("ThermalAnomalyInference initialized successfully")
    
    def _load_model(self, model_path: str):
        """Load trained model weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded model weights from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            raise
    
    def preprocess_input(self, input_data: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess input data for inference.
        
        Args:
            input_data: Path to thermal data file or numpy/tensor array
            
        Returns:
            Preprocessed tensor ready for inference
        """
        if isinstance(input_data, str):
            # Load thermal data from file
            thermal_data = load_thermal_file(
                input_data,
                normalize=self.config['dataset']['normalize'],
                target_size=tuple(self.config['dataset']['image_size']),
                temperature_range=tuple(self.config['dataset']['temperature_range'])
            )
            x = torch.from_numpy(thermal_data['thermal_data'])
        elif isinstance(input_data, np.ndarray):
            x = torch.from_numpy(input_data)
        else:
            x = input_data
            
        # Add batch and channel dimensions if needed
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
            
        return x.to(self.device, dtype=torch.float32)
    
    def postprocess_output(self, output: torch.Tensor) -> np.ndarray:
        """
        Postprocess model output.
        
        Args:
            output: Raw model output tensor
            
        Returns:
            Processed numpy array
        """
        with torch.no_grad():
            # Apply sigmoid if not already applied
            if self.config['model'].get('apply_sigmoid', True):
                output = torch.sigmoid(output)
            
            # Convert to numpy
            output = output.cpu().numpy()
            
            # Squeeze extra dimensions
            output = np.squeeze(output)
            
            # Apply threshold if specified
            threshold = self.config['inference'].get('threshold', 0.5)
            if self.config['inference'].get('apply_threshold', False):
                output = (output > threshold).astype(np.float32)
            
            return output
    
    def run_inference(self, input_data: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Path to thermal data file or numpy/tensor array
            
        Returns:
            Anomaly prediction map as numpy array
        """
        # Preprocess input
        x = self.preprocess_input(input_data)
        
        # Run inference
        with torch.no_grad():
            output = self.model(x)
        
        # Postprocess output
        return self.postprocess_output(output)
    
    def batch_inference(self, input_dir: str) -> Dict[str, np.ndarray]:
        """
        Run inference on all thermal files in a directory.
        
        Args:
            input_dir: Directory containing thermal data files
            
        Returns:
            Dictionary mapping filenames to prediction maps
        """
        input_dir = Path(input_dir)
        results = {}
        
        # Get list of thermal files
        thermal_files = list(input_dir.glob("*_ST_B10.TIF"))
        
        for file_path in thermal_files:
            try:
                logger.info(f"Processing {file_path.name}")
                prediction = self.run_inference(str(file_path))
                results[file_path.name] = prediction
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                continue
        
        return results
    
    def generate_submission(self, 
                          prediction_map: np.ndarray,
                          input_path: str,
                          metrics: Optional[Dict[str, float]] = None) -> Dict[str, str]:
        """
        Generate submission files for a prediction.
        
        Args:
            prediction_map: Anomaly prediction map
            input_path: Path to input thermal file
            metrics: Optional dictionary of performance metrics
            
        Returns:
            Dictionary with paths to generated files
        """
        if metrics is None:
            metrics = {}
            
        return self.submission_gen.generate_all_outputs(
            prediction_map=prediction_map,
            input_path=input_path,
            metrics=metrics,
            model_path=self.config['model'].get('weights_path', 'None'),
            model_name=self.config['model'].get('name', 'SwinUNet')
        )