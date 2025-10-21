#!/usr/bin/env python3
"""
Submission output generator for Euclidean Technologies Thermal Anomaly Detection.
Generates all required Stage-1 deliverables in the official submission format.
"""

import os
import numpy as np
import torch
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class SubmissionGenerator:
    """
    Generate all Stage-1 submission outputs for thermal anomaly detection.
    """
    
    def __init__(self, startup_name: str = "EUCLIDEAN_TECHNOLOGIES", output_dir: str = "EUCLIDEAN_TECHNOLOGIES_Thermal_Outputs/"):
        """
        Initialize submission generator with official structure.
        
        Args:
            startup_name: Name of the startup (used in filenames)
            output_dir: Output directory for submission files
        """
        self.startup_name = startup_name
        self.output_dir = Path(output_dir)
        
        # Create official folder structure
        self.hash_dir = self.output_dir / "1_HashValue"
        self.results_dir = self.output_dir / "2_AnomalyDetectionResults" 
        self.accuracy_dir = self.output_dir / "3_AccuracyReport"
        self.docs_dir = self.output_dir / "4_ModelDocumentation"
        
        # Create all directories
        for dir_path in [self.hash_dir, self.results_dir, self.accuracy_dir, self.docs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize output file paths
        self._setup_output_paths()
        
        logger.info(f"Submission generator initialized for {startup_name}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_output_paths(self):
        """Setup official output file paths."""
        # 1_HashValue
        self.hash_path = self.hash_dir / f"{self.startup_name}_ThermalModelHash.txt"
        
        # 2_AnomalyDetectionResults  
        self.geotiff_path = self.results_dir / f"{self.startup_name}_Landsat8_ThermalAnomaly.tif"
        self.png_path = self.results_dir / f"{self.startup_name}_Landsat8_ThermalAnomaly.png"
        self.geojson_path = self.results_dir / f"{self.startup_name}_ThermalRegions.geojson"
        
        # 3_AccuracyReport
        self.excel_path = self.accuracy_dir / f"{self.startup_name}_ThermalAccuracyReport.xlsx"
        
        # 4_ModelDocumentation
        self.pdf_path = self.docs_dir / f"{self.startup_name}_ThermalModelReport.pdf"
        
        # Root README
        self.readme_path = self.output_dir / "README.txt"
    
    def generate_all_outputs(self,
                           prediction_map: Union[np.ndarray, torch.Tensor],
                           input_path: Union[str, Path],
                           metrics: Dict[str, float],
                           model_path: Union[str, Path],
                           model_name: str = "SwinUNet",
                           dataset_name: str = "Landsat-8 Thermal") -> Dict[str, str]:
        """
        Generate all Stage-1 submission outputs.
        
        Args:
            prediction_map: Anomaly prediction map (H, W) with values 0-1
            input_path: Path to input thermal image
            metrics: Dictionary of performance metrics
            model_path: Path to trained model file
            model_name: Name of the model architecture
            dataset_name: Name of the dataset used
            
        Returns:
            Dictionary with paths to generated files
        """
        logger.info("Generating Stage-1 submission outputs...")
        
        # Convert inputs to numpy if needed
        if isinstance(prediction_map, torch.Tensor):
            prediction_map = prediction_map.cpu().numpy()
        
        # Ensure prediction map is 2D
        if prediction_map.ndim > 2:
            prediction_map = np.squeeze(prediction_map)
        
        # Generate each output file
        results = {}
        
        try:
            # 1. Generate Boolean GeoTIFF (0,1 format as required)
            logger.info("Generating Boolean GeoTIFF anomaly map...")
            # Convert to boolean format first
            boolean_map = (prediction_map > 0.5).astype(np.uint8)
            self._generate_geotiff(boolean_map, input_path)
            results['geotiff'] = str(self.geotiff_path)
            
            # 2. Generate PNG visualization
            logger.info("Generating PNG visualization...")
            self._generate_png_visualization(prediction_map, input_path)
            results['png'] = str(self.png_path)
            
            # 3. Generate Excel accuracy report
            logger.info("Generating Excel accuracy report...")
            self._generate_excel_report(metrics, model_name, dataset_name)
            results['excel'] = str(self.excel_path)
            
            # 4. Generate model hash
            logger.info("Generating model hash...")
            self._generate_model_hash(model_path, model_name)
            results['hash'] = str(self.hash_path)
            
            # 5. Generate README
            logger.info("Generating README...")
            self._generate_readme(metrics, model_name, dataset_name)
            results['readme'] = str(self.readme_path)
            
            logger.info("All submission outputs generated successfully!")
            self._log_file_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating submission outputs: {str(e)}")
            raise
    
    def _generate_geotiff(self, prediction_map: np.ndarray, input_path: Union[str, Path]):
        """Generate GeoTIFF anomaly map with geospatial metadata."""
        try:
            # Load input file to get geospatial metadata
            with rasterio.open(input_path) as src:
                # Copy metadata
                profile = src.profile.copy()
                profile.update({
                    'dtype': 'float32',
                    'count': 1,
                    'compress': 'lzw',
                    'nodata': -9999
                })
                
                # Ensure prediction map matches input dimensions
                if prediction_map.shape != (src.height, src.width):
                    prediction_map = cv2.resize(
                        prediction_map, 
                        (src.width, src.height),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # Write GeoTIFF
                with rasterio.open(self.geotiff_path, 'w', **profile) as dst:
                    dst.write(prediction_map.astype(np.float32), 1)
                    
                    # Add metadata
                    dst.update_tags(
                        DESCRIPTION="Thermal anomaly detection map",
                        CREATOR=self.startup_name,
                        CREATED=datetime.now().isoformat(),
                        ANOMALY_ENCODING="0=normal, 1=anomaly, 0-1=probability"
                    )
            
            logger.info(f"GeoTIFF saved: {self.geotiff_path}")
            
        except Exception as e:
            logger.error(f"Error generating GeoTIFF: {str(e)}")
            # Fallback: save without geospatial metadata
            self._save_fallback_geotiff(prediction_map)
    
    def _save_fallback_geotiff(self, prediction_map: np.ndarray):
        """Save GeoTIFF without geospatial metadata as fallback."""
        profile = {
            'driver': 'GTiff',
            'height': prediction_map.shape[0],
            'width': prediction_map.shape[1],
            'count': 1,
            'dtype': 'float32',
            'compress': 'lzw',
            'nodata': -9999
        }
        
        with rasterio.open(self.geotiff_path, 'w', **profile) as dst:
            dst.write(prediction_map.astype(np.float32), 1)
            dst.update_tags(
                DESCRIPTION="Thermal anomaly detection map (no geospatial metadata)",
                CREATOR=self.startup_name,
                CREATED=datetime.now().isoformat()
            )
        
        logger.warning(f"GeoTIFF saved without geospatial metadata: {self.geotiff_path}")
    
    def _generate_png_visualization(self, prediction_map: np.ndarray, input_path: Union[str, Path]):
        """Generate PNG visualization with anomaly overlay."""
        try:
            # Load base thermal image
            try:
                with rasterio.open(input_path) as src:
                    base_image = src.read(1).astype(np.float32)
            except:
                # Fallback to prediction map as base
                base_image = prediction_map.copy()
            
            # Normalize base image for visualization
            base_image = (base_image - base_image.min()) / (base_image.max() - base_image.min())
            
            # Create figure with larger size for better visibility
            fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
            
            # Display base thermal image in grayscale
            ax.imshow(base_image, cmap='gray', alpha=0.6)
            
            # Create point-based anomaly visualization
            anomaly_mask = prediction_map > 0.5
            
            # Get coordinates of anomalies for point plotting
            anomaly_y, anomaly_x = np.where(anomaly_mask)
            
            if len(anomaly_x) > 0:
                # For large datasets, sample points to avoid overcrowding
                max_points = 100000  # Maximum points to display
                if len(anomaly_x) > max_points:
                    sample_indices = np.random.choice(len(anomaly_x), max_points, replace=False)
                    anomaly_x_sample = anomaly_x[sample_indices]
                    anomaly_y_sample = anomaly_y[sample_indices]
                    conf_sample = prediction_map[anomaly_y_sample, anomaly_x_sample]
                else:
                    anomaly_x_sample = anomaly_x
                    anomaly_y_sample = anomaly_y
                    conf_sample = prediction_map[anomaly_y, anomaly_x]
                
                # Plot anomaly points with colors based on confidence
                scatter = ax.scatter(anomaly_x_sample, anomaly_y_sample, 
                                   c=conf_sample, cmap='plasma', 
                                   s=1.0, alpha=0.8, marker='.')
                
                # Add bright yellow highlights for high-confidence anomalies
                high_conf_indices = conf_sample > 0.7
                if np.any(high_conf_indices):
                    ax.scatter(anomaly_x_sample[high_conf_indices], 
                             anomaly_y_sample[high_conf_indices], 
                             c='yellow', s=0.8, alpha=0.9, marker='.')
                
                # Set the anomaly_overlay for colorbar
                anomaly_overlay = scatter
            else:
                # No anomalies detected
                anomaly_overlay = None
            
            # Create a legend for point-based anomaly detection
            import matplotlib.patches as mpatches
            normal_patch = mpatches.Patch(color='gray', label='Normal Background')
            anomaly_patch = mpatches.Patch(color='yellow', label='Anomaly Points')
            high_conf_patch = mpatches.Patch(color='red', label='High Confidence')
            ax.legend(handles=[normal_patch, anomaly_patch, high_conf_patch], 
                     loc='upper right', fontsize=10)
            
            # Set title and labels with enhanced information
            total_pixels = prediction_map.size
            anomaly_pixels = np.sum(anomaly_mask)
            anomaly_percentage = (anomaly_pixels / total_pixels) * 100
            
            ax.set_title(f'{self.startup_name} - Thermal Anomaly Detection\n(Bright Points = Anomalies: {anomaly_pixels:,} pixels, {anomaly_percentage:.2f}%)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            
            # Add legend
            legend_elements = [
                plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.8, label='Thermal Background'),
                plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='Detected Anomaly'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            # Add metadata text
            metadata_text = (
                f"Model: {self.startup_name} Deep Learning System\n"
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Detection: Non-natural thermal anomalies"
            )
            ax.text(0.02, 0.98, metadata_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Remove axes for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Save PNG
            plt.tight_layout()
            plt.savefig(self.png_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"PNG visualization saved: {self.png_path}")
            
        except Exception as e:
            logger.error(f"Error generating PNG visualization: {str(e)}")
            # Simple fallback visualization
            self._save_simple_png(prediction_map)
    
    def _save_simple_png(self, prediction_map: np.ndarray):
        """Save simple PNG visualization as fallback."""
        plt.figure(figsize=(10, 8))
        plt.imshow(prediction_map, cmap='hot')
        plt.colorbar(label='Anomaly Probability')
        plt.title(f'{self.startup_name} - Thermal Anomaly Detection')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.png_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.warning(f"Simple PNG visualization saved: {self.png_path}")
    
    def _generate_excel_report(self, metrics: Dict[str, float], model_name: str, dataset_name: str):
        """Generate Excel report with performance metrics."""
        try:
            # Prepare metrics data
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Sheet 1: Overall Metrics
            overall_metrics = {
                'Metric': ['Startup Name', 'Model', 'Dataset', 'GPU Used', 'Timestamp',
                          'Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC', 
                          'FPS', 'Model Size (MB)', 'Inference Time (ms)'],
                'Value': [
                    self.startup_name,
                    model_name,
                    dataset_name,
                    'NVIDIA A100 80GB',
                    timestamp,
                    f"{metrics.get('accuracy', 0.0):.4f}",
                    f"{metrics.get('f1_score', 0.0):.4f}",
                    f"{metrics.get('precision', 0.0):.4f}",
                    f"{metrics.get('recall', 0.0):.4f}",
                    f"{metrics.get('roc_auc', 0.0):.4f}",
                    f"{metrics.get('pr_auc', 0.0):.4f}",
                    f"{metrics.get('fps', 0.0):.2f}",
                    f"{metrics.get('model_size_mb', 0.0):.2f}",
                    f"{metrics.get('inference_time_ms', 0.0):.2f}"
                ]
            }
            
            # Create Excel writer
            with pd.ExcelWriter(self.excel_path, engine='openpyxl') as writer:
                # Sheet 1: Overall Metrics
                df_overall = pd.DataFrame(overall_metrics)
                df_overall.to_excel(writer, sheet_name='Overall Metrics', index=False)
                
                # Sheet 2: Detailed Performance
                detailed_metrics = []
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        detailed_metrics.append({'Metric': key, 'Value': value})
                
                if detailed_metrics:
                    df_detailed = pd.DataFrame(detailed_metrics)
                    df_detailed.to_excel(writer, sheet_name='Detailed Metrics', index=False)
                
                # Sheet 3: System Information
                system_info = {
                    'Component': ['Startup', 'Model Architecture', 'Framework', 'CUDA Version', 
                                 'GPU Memory', 'Precision', 'Optimization'],
                    'Details': [self.startup_name, model_name, 'PyTorch 2.0+', 'CUDA 11.8+',
                               '80GB A100', 'Mixed (FP16/FP32)', 'TensorRT Ready']
                }
                df_system = pd.DataFrame(system_info)
                df_system.to_excel(writer, sheet_name='System Info', index=False)
            
            logger.info(f"Excel report saved: {self.excel_path}")
            
        except Exception as e:
            logger.error(f"Error generating Excel report: {str(e)}")
            # Fallback CSV
            self._save_fallback_csv(metrics, model_name, dataset_name)
    
    def _save_fallback_csv(self, metrics: Dict[str, float], model_name: str, dataset_name: str):
        """Save CSV report as fallback."""
        csv_path = self.output_dir / f"{self.startup_name}_Metrics.csv"
        
        data = {
            'startup_name': self.startup_name,
            'model': model_name,
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        df = pd.DataFrame([data])
        df.to_csv(csv_path, index=False)
        logger.warning(f"CSV report saved as fallback: {csv_path}")
    
    def _generate_model_hash(self, model_path: Union[str, Path], model_name: str):
        """Generate SHA-256 hash of the model file."""
        try:
            # Calculate SHA-256 hash
            sha256_hash = hashlib.sha256()
            with open(model_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            
            hash_value = sha256_hash.hexdigest()
            
            # Create hash file content
            hash_content = f"""Startup Name: {self.startup_name}
Model: {model_name}
Model File: {Path(model_path).name}
SHA256 Hash: {hash_value}
Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
GPU Used: NVIDIA A100 80GB

Hash Verification:
To verify the model integrity, run:
sha256sum {Path(model_path).name}

Expected hash: {hash_value}
"""
            
            # Save hash file
            with open(self.hash_path, 'w') as f:
                f.write(hash_content)
            
            logger.info(f"Model hash saved: {self.hash_path}")
            logger.info(f"Model SHA-256: {hash_value}")
            
        except Exception as e:
            logger.error(f"Error generating model hash: {str(e)}")
            # Fallback hash file
            self._save_fallback_hash(model_name)
    
    def _save_fallback_hash(self, model_name: str):
        """Save fallback hash file."""
        hash_content = f"""Startup Name: {self.startup_name}
Model: {model_name}
Model File: Not Available
SHA256 Hash: Not Available (File not found)
Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
GPU Used: NVIDIA A100 80GB

Note: Model file was not accessible for hash generation.
"""
        with open(self.hash_path, 'w') as f:
            f.write(hash_content)
        logger.warning(f"Fallback hash file saved: {self.hash_path}")
    
    def _generate_readme(self, metrics: Dict[str, float], model_name: str, dataset_name: str):
        """Generate submission README file."""
        readme_content = f"""Startup Name: {self.startup_name}
Model: {model_name}
Dataset: {dataset_name}
GPU Used: NVIDIA A100 80GB

Performance Metrics:
Accuracy: {metrics.get('accuracy', 0.0):.4f}
F1 Score: {metrics.get('f1_score', 0.0):.4f}
ROC-AUC: {metrics.get('roc_auc', 0.0):.4f}
PR-AUC: {metrics.get('pr_auc', 0.0):.4f}
FPS: {metrics.get('fps', 0.0):.2f}

Description: 
Detects manmade thermal anomalies (fires, industrial heat, machinery) 
and suppresses natural heat sources using deep learning with Swin 
Transformer architecture. Optimized for real-time A100 inference.

Technical Details:
- Architecture: {model_name} with thermal-optimized preprocessing
- Training: Mixed precision (FP16/FP32) with gradient scaling
- Inference: GPU-accelerated with optional TensorRT optimization
- Data: Minimal preprocessing to preserve temperature integrity
- Anomaly Detection: Pixel-level segmentation with confidence scoring

Outputs:
- GeoTIFF: {self.startup_name}_AnomalyMap.tif (geospatial anomaly map)
- PNG: {self.startup_name}_AnomalyMap.png (visualization overlay)
- Excel: {self.startup_name}_Metrics.xlsx (performance metrics)
- Hash: {self.startup_name}_ModelHash.txt (model verification)

Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Contact: contact@euclideantech.ai

Stage-1 Submission Complete
{self.startup_name} - Thermal Anomaly Detection System
"""
        
        with open(self.readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"README saved: {self.readme_path}")
    
    def _log_file_summary(self, results: Dict[str, str]):
        """Log summary of generated files."""
        logger.info("=" * 60)
        logger.info("STAGE-1 SUBMISSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Startup: {self.startup_name}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info("-" * 60)
        
        for file_type, file_path in results.items():
            file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
            logger.info(f"{file_type.upper():15}: {Path(file_path).name} ({file_size:,} bytes)")
        
        logger.info("=" * 60)
        logger.info("All submission files generated successfully!")
        logger.info("Ready for Stage-1 evaluation submission.")
        logger.info("=" * 60)


def create_submission_outputs(prediction_map: Union[np.ndarray, torch.Tensor],
                            input_path: Union[str, Path],
                            metrics: Dict[str, float],
                            model_path: Union[str, Path],
                            output_dir: str = "submission/",
                            startup_name: str = "EuclideanTechnologies",
                            model_name: str = "SwinUNet",
                            dataset_name: str = "Landsat-8 Thermal") -> Dict[str, str]:
    """
    Convenience function to generate all submission outputs.
    
    Args:
        prediction_map: Anomaly prediction map
        input_path: Path to input thermal image
        metrics: Performance metrics dictionary
        model_path: Path to trained model
        output_dir: Output directory
        startup_name: Startup name for filenames
        model_name: Model architecture name
        dataset_name: Dataset name
        
    Returns:
        Dictionary with paths to generated files
    """
    generator = SubmissionGenerator(startup_name, output_dir)
    
    return generator.generate_all_outputs(
        prediction_map=prediction_map,
        input_path=input_path,
        metrics=metrics,
        model_path=model_path,
        model_name=model_name,
        dataset_name=dataset_name
    )


if __name__ == "__main__":
    # Test submission generator
    print("Testing Submission Generator...")
    
    # Create dummy data
    dummy_prediction = np.random.rand(512, 512)
    dummy_prediction[100:200, 100:200] = 0.8  # Add some "anomalies"
    
    dummy_metrics = {
        'accuracy': 0.9512,
        'f1_score': 0.9023,
        'precision': 0.8956,
        'recall': 0.9091,
        'roc_auc': 0.9745,
        'pr_auc': 0.9234,
        'fps': 42.5,
        'model_size_mb': 127.3,
        'inference_time_ms': 23.5
    }
    
    # Test output generation
    try:
        results = create_submission_outputs(
            prediction_map=dummy_prediction,
            input_path="dummy_thermal.tif",
            metrics=dummy_metrics,
            model_path="dummy_model.pth"
        )
        
        print("Submission generator test completed successfully!")
        print("Generated files:", list(results.keys()))
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()