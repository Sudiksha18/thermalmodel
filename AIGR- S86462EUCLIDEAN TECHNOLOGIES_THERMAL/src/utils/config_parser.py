#!/usr/bin/env python3
"""
Configuration parser for thermal anomaly detection system.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(save_path, 'w') as f:
            if save_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif save_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {save_path.suffix}")
        
        logger.info(f"Saved configuration to {save_path}")
        
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        raise


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and set defaults for configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration
    """
    # Default configuration
    default_config = {
        'model': {
            'name': 'SwinUNet',
            'backbone': 'swin_tiny',
            'num_classes': 2,
            'input_channels': 1,
            'pretrained': True,
            'dropout': 0.1
        },
        'dataset': {
            'name': 'thermal_anomaly',
            'data_dir': 'data/',
            'image_size': [512, 512],
            'normalize': True,
            'augmentation': True,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'formats': ['.tif', '.he5', '.png', '.jpg']
        },
        'training': {
            'epochs': 100,
            'batch_size': 8,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'scheduler': 'cosine',
            'use_amp': True,
            'gradient_clip': 1.0,
            'loss_function': 'combined',
            'patience': 15,
            'min_delta': 1e-4
        },
        'validation': {
            'val_frequency': 5,
            'save_best': True,
            'metric': 'f1_score'
        },
        'inference': {
            'batch_size': 16,
            'tta': False,
            'threshold': 0.5,
            'save_probability': True,
            'save_overlay': True,
            'overlay_alpha': 0.6
        },
        'hardware': {
            'device': 'cuda',
            'gpu_ids': [0],
            'num_workers': 8,
            'pin_memory': True,
            'mixed_precision': True,
            'benchmark': True
        },
        'logging': {
            'log_dir': 'outputs/logs/',
            'experiment_name': 'thermal_anomaly_detection',
            'save_frequency': 10,
            'log_images': True,
            'max_images': 8,
            'use_wandb': False
        },
        'export': {
            'startup_name': 'EuclideanTechnologies',
            'output_dir': 'submission/',
            'geotiff_compression': 'lzw',
            'png_dpi': 300,
            'excel_engine': 'openpyxl',
            'author': 'Euclidean Technologies',
            'description': 'Deep learning thermal anomaly detection',
            'contact': 'contact@euclideantech.ai'
        },
        'evaluation': {
            'metrics': ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc', 'pr_auc'],
            'confidence_threshold': [0.3, 0.5, 0.7],
            'iou_threshold': 0.5,
            'plot_confusion_matrix': True,
            'plot_roc_curve': True,
            'plot_pr_curve': True,
            'save_predictions': True
        }
    }
    
    # Merge with defaults
    validated_config = deep_merge(default_config, config)
    
    # Validate required fields
    required_fields = [
        'model.name',
        'dataset.data_dir',
        'training.batch_size',
        'training.learning_rate'
    ]
    
    for field in required_fields:
        if not get_nested_value(validated_config, field):
            raise ValueError(f"Required configuration field missing: {field}")
    
    logger.info("Configuration validated successfully")
    return validated_config


def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def get_nested_value(config: Dict, key_path: str) -> Any:
    """
    Get nested value from configuration using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'model.name')
        
    Returns:
        Value at key path or None if not found
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    
    return value


def set_nested_value(config: Dict, key_path: str, value: Any) -> None:
    """
    Set nested value in configuration using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = load_config("config.yaml")
        validated_config = validate_config(config)
        print("Configuration loaded and validated successfully")
        print(f"Model: {get_nested_value(validated_config, 'model.name')}")
        print(f"Batch size: {get_nested_value(validated_config, 'training.batch_size')}")
    except Exception as e:
        print(f"Configuration test failed: {e}")