#!/usr/bin/env python3
"""
Logging utilities for thermal anomaly detection system.
"""

import os
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import colorlog


def setup_logger(logging_config: Dict, debug: bool = False) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        logging_config: Logging configuration dictionary
        debug: Enable debug logging
        
    Returns:
        Configured logger
    """
    # Create logs directory
    log_dir = Path(logging_config.get('log_dir', 'outputs/logs/'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set logging level
    level = logging.DEBUG if debug else logging.INFO
    
    # Create logger
    logger = logging.getLogger('thermal_anomaly')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = logging_config.get('experiment_name', 'thermal_anomaly_detection')
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Log initial message
    logger.info(f"Logger initialized - Log file: {log_file}")
    logger.info(f"Logging level: {logging.getLevelName(level)}")
    
    return logger


def create_checkpoint_logger(checkpoint_dir: Path) -> logging.Logger:
    """
    Create logger for checkpoint information.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Checkpoint logger
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('checkpoints')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler for checkpoints
    checkpoint_log = checkpoint_dir / "checkpoints.log"
    file_handler = logging.FileHandler(checkpoint_log)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    )
    logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """Logger for training metrics and losses."""
    
    def __init__(self, log_dir: Path, experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.csv"
        
        # Initialize CSV file
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w') as f:
                f.write("epoch,phase,loss,accuracy,f1_score,precision,recall,roc_auc,pr_auc,learning_rate\n")
    
    def log_metrics(self, epoch: int, phase: str, metrics: Dict, learning_rate: float = None):
        """
        Log metrics to CSV file.
        
        Args:
            epoch: Current epoch
            phase: Training phase (train/val/test)
            metrics: Dictionary of metrics
            learning_rate: Current learning rate
        """
        try:
            with open(self.metrics_file, 'a') as f:
                loss = metrics.get('loss', 0.0)
                accuracy = metrics.get('accuracy', 0.0)
                f1_score = metrics.get('f1_score', 0.0)
                precision = metrics.get('precision', 0.0)
                recall = metrics.get('recall', 0.0)
                roc_auc = metrics.get('roc_auc', 0.0)
                pr_auc = metrics.get('pr_auc', 0.0)
                lr = learning_rate or 0.0
                
                f.write(f"{epoch},{phase},{loss:.6f},{accuracy:.4f},{f1_score:.4f},"
                       f"{precision:.4f},{recall:.4f},{roc_auc:.4f},{pr_auc:.4f},{lr:.8f}\n")
        
        except Exception as e:
            logging.getLogger('thermal_anomaly').error(f"Error logging metrics: {str(e)}")
    
    def log_epoch_summary(self, epoch: int, train_metrics: Dict, val_metrics: Dict, 
                         learning_rate: float, epoch_time: float):
        """
        Log epoch summary to main logger.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
            learning_rate: Current learning rate
            epoch_time: Time taken for epoch
        """
        logger = logging.getLogger('thermal_anomaly')
        
        logger.info(f"Epoch {epoch:03d} Summary:")
        logger.info(f"  Time: {epoch_time:.2f}s, LR: {learning_rate:.2e}")
        
        # Training metrics
        logger.info(f"  Train - Loss: {train_metrics.get('loss', 0):.4f}, "
                   f"F1: {train_metrics.get('f1_score', 0):.4f}, "
                   f"Acc: {train_metrics.get('accuracy', 0):.4f}")
        
        # Validation metrics
        logger.info(f"  Val   - Loss: {val_metrics.get('loss', 0):.4f}, "
                   f"F1: {val_metrics.get('f1_score', 0):.4f}, "
                   f"Acc: {val_metrics.get('accuracy', 0):.4f}")


class WandbLogger:
    """Weights & Biases logger integration."""
    
    def __init__(self, config: Dict, enabled: bool = True):
        self.enabled = enabled and config.get('logging', {}).get('use_wandb', False)
        
        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
                
                # Initialize wandb
                wandb.init(
                    project=config.get('logging', {}).get('wandb_project', 'thermal-anomaly-detection'),
                    entity=config.get('logging', {}).get('wandb_entity', None),
                    config=config,
                    name=config.get('logging', {}).get('experiment_name', 'experiment'),
                    tags=['thermal', 'anomaly-detection', 'pytorch']
                )
                
                logging.getLogger('thermal_anomaly').info("Weights & Biases logging enabled")
                
            except ImportError:
                logging.getLogger('thermal_anomaly').warning("wandb not installed, disabling wandb logging")
                self.enabled = False
            except Exception as e:
                logging.getLogger('thermal_anomaly').error(f"Failed to initialize wandb: {str(e)}")
                self.enabled = False
        else:
            self.wandb = None
    
    def log_metrics(self, metrics: Dict, step: int):
        """Log metrics to wandb."""
        if self.enabled and self.wandb:
            self.wandb.log(metrics, step=step)
    
    def log_images(self, images: Dict, step: int):
        """Log images to wandb."""
        if self.enabled and self.wandb:
            wandb_images = {}
            for key, image in images.items():
                if hasattr(image, 'numpy'):
                    image = image.numpy()
                wandb_images[key] = self.wandb.Image(image)
            self.wandb.log(wandb_images, step=step)
    
    def finish(self):
        """Finish wandb run."""
        if self.enabled and self.wandb:
            self.wandb.finish()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name (default: 'thermal_anomaly')
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name or 'thermal_anomaly')


if __name__ == "__main__":
    # Test logging setup
    logging_config = {
        'log_dir': 'test_logs/',
        'experiment_name': 'test_experiment',
        'use_wandb': False
    }
    
    logger = setup_logger(logging_config, debug=True)
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test metrics logger
    metrics_logger = MetricsLogger(Path('test_logs'), 'test_experiment')
    metrics_logger.log_metrics(
        epoch=1,
        phase='train',
        metrics={'loss': 0.5, 'accuracy': 0.85, 'f1_score': 0.82},
        learning_rate=1e-4
    )
    
    print("Logging test completed!")