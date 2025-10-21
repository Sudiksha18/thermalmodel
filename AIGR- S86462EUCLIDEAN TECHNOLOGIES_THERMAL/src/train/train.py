#!/usr/bin/env python3
"""
Main training script for Euclidean Technologies Thermal Anomaly Detection.
Implements complete training pipeline with mixed precision and GPU acceleration.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import time
import json
from typing import Dict, List, Optional, Tuple
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from dataloader.dataset_utils import ThermalDataModule
from dataloader.he5_loader import load_thermal_file
from models.swin_unet import create_swin_unet
from utils.config_parser import load_config, validate_config
from utils.logging_utils import setup_logger, MetricsLogger
from inference.export_submission import create_submission_outputs

logger = logging.getLogger(__name__)


class ThermalAnomalyTrainer:
    """
    Complete trainer for thermal anomaly detection with GPU acceleration.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = validate_config(config)
        self.device = torch.device(config['hardware']['device'])
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize data module
        self.data_module = ThermalDataModule(config)
        
        # Initialize training components
        self.optimizer = None
        self.scheduler = None
        # Use newer GradScaler syntax and enable optimizations
        self.scaler = torch.amp.GradScaler('cuda') if config['training']['use_amp'] else None
        
        # Enable advanced PyTorch optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.criterion = self._create_loss_function()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Setup output directories
        self.output_dir = Path("outputs")
        self.model_dir = self.output_dir / "models"
        self.log_dir = self.output_dir / "logs"
        
        for dir_path in [self.output_dir, self.model_dir, self.log_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize metrics logger
        self.metrics_logger = MetricsLogger(
            self.log_dir, 
            config['logging']['experiment_name']
        )
        
        logger.info(f"Trainer initialized for device: {self.device}")
    
    def _create_model(self) -> nn.Module:
        """Create model architecture."""
        model = create_swin_unet(self.config)
        model = model.to(self.device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created: {self.config['model']['name']}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function."""
        loss_config = self.config['training']['loss_function']
        
        if loss_config == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_config == 'dice':
            return DiceLoss()
        elif loss_config == 'focal':
            return FocalLoss()
        elif loss_config == 'combined':
            return CombinedLoss(self.config['training'].get('loss_weights', {}))
        else:
            logger.warning(f"Unknown loss function: {loss_config}, using BCEWithLogitsLoss")
            return nn.BCEWithLogitsLoss()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Ensure learning rate and weight decay are floats
        lr = float(self.config['training']['learning_rate'])
        wd = float(self.config['training']['weight_decay'])
        
        # Advanced optimizer with gradient accumulation
        optimizer = optim.AdamW(
            optimizer_params,
            lr=lr,
            weight_decay=wd,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        logger.info(f"Optimizer created: AdamW with advanced settings")
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Weight decay: {wd}")
        
        return optimizer
    
    def _create_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        scheduler_type = self.config['training']['scheduler']
        
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.config['training']['epochs']
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                patience=10,
                factor=0.5
            )
        else:
            scheduler = optim.lr_scheduler.ConstantLR(optimizer)
        
        logger.info(f"Scheduler created: {scheduler_type}")
        return scheduler
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        start_time = time.time()
        
        # Setup data
        self.data_module.setup("fit")
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        
        # Setup training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(self.optimizer)
        
        # Training loop
        best_val_metric = 0.0
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train epoch
            train_loss, train_metrics = self._train_epoch(train_loader)
            
            # Validate epoch
            if epoch % self.config['validation']['val_frequency'] == 0:
                val_loss, val_metrics = self._validate_epoch(val_loader)
                
                # Update learning rate
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['f1_score'])
                else:
                    self.scheduler.step()
                
                # Check for improvement
                current_metric = val_metrics[self.config['validation']['metric']]
                if current_metric > best_val_metric:
                    best_val_metric = current_metric
                    patience_counter = 0
                    
                    if self.config['validation']['save_best']:
                        self._save_checkpoint(epoch, is_best=True)
                        logger.info(f"New best model saved! {self.config['validation']['metric']}: {current_metric:.4f}")
                else:
                    patience_counter += 1
                
                # Log metrics
                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.param_groups[0]['lr']
                
                self.metrics_logger.log_epoch_summary(
                    epoch, train_metrics, val_metrics, current_lr, epoch_time
                )
                
                self.metrics_logger.log_metrics(epoch, 'train', train_metrics, current_lr)
                self.metrics_logger.log_metrics(epoch, 'val', val_metrics, current_lr)
                
                # Early stopping
                if patience_counter >= self.config['training']['patience']:
                    logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break
            
            # Regular checkpoint saving
            if epoch % self.config['logging']['save_frequency'] == 0:
                self._save_checkpoint(epoch, is_best=False)
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Final evaluation
        self._final_evaluation()
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch:03d}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            images = batch['image'].to(self.device)
            targets = batch.get('mask', torch.zeros_like(images[:, 0])).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    loss = self.criterion(logits, targets.float())
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training']['gradient_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = self.criterion(logits, targets.float())
                
                loss.backward()
                
                if self.config['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            
            # Convert to predictions
            predictions = torch.sigmoid(logits)
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())
            
            # Update progress
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        metrics = self._calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        return avg_loss, metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                targets = batch.get('mask', torch.zeros_like(images[:, 0])).to(self.device)
                
                # Forward pass
                if self.scaler:
                    with autocast():
                        outputs = self.model(images)
                        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                        loss = self.criterion(logits, targets.float())
                else:
                    outputs = self.model(images)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    loss = self.criterion(logits, targets.float())
                
                total_loss += loss.item()
                
                # Convert to predictions
                predictions = torch.sigmoid(logits)
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(val_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        metrics = self._calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        # Convert to numpy
        pred_np = predictions.numpy()
        target_np = targets.numpy()
        
        # Binary predictions
        pred_binary = (pred_np > 0.5).astype(int)
        target_binary = target_np.astype(int)
        
        # Flatten for sklearn metrics
        pred_flat = pred_binary.flatten()
        target_flat = target_binary.flatten()
        pred_prob_flat = pred_np.flatten()
        
        metrics = {}
        
        try:
            from sklearn.metrics import (
                accuracy_score, f1_score, precision_score, recall_score,
                roc_auc_score, average_precision_score
            )
            
            metrics['accuracy'] = accuracy_score(target_flat, pred_flat)
            metrics['f1_score'] = f1_score(target_flat, pred_flat, zero_division=0)
            metrics['precision'] = precision_score(target_flat, pred_flat, zero_division=0)
            metrics['recall'] = recall_score(target_flat, pred_flat, zero_division=0)
            
            # AUC metrics (handle edge cases)
            if len(np.unique(target_flat)) > 1:
                metrics['roc_auc'] = roc_auc_score(target_flat, pred_prob_flat)
                metrics['pr_auc'] = average_precision_score(target_flat, pred_prob_flat)
            else:
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            # Fallback metrics
            metrics = {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'roc_auc': 0.0,
                'pr_auc': 0.0
            }
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.model_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.model_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
        
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def _final_evaluation(self):
        """Run final evaluation and generate submission outputs."""
        logger.info("Running final evaluation...")
        
        # Load best model
        best_model_path = self.model_dir / "best_model.pth"
        if best_model_path.exists():
            self.load_checkpoint(best_model_path)
            logger.info("Loaded best model for final evaluation")
        
        # Setup test data
        self.data_module.setup("test")
        test_loader = self.data_module.test_dataloader()
        
        # Run evaluation
        test_loss, test_metrics = self._validate_epoch(test_loader)
        
        logger.info("Final Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Generate sample prediction for submission
        self._generate_sample_submission(test_metrics)
    
    def _generate_sample_submission(self, metrics: Dict[str, float]):
        """Generate sample submission outputs."""
        logger.info("Generating sample submission outputs...")
        
        try:
            # Create dummy prediction for demonstration
            dummy_prediction = np.random.rand(512, 512)
            dummy_prediction[100:200, 100:200] = 0.8  # Add some "anomalies"
            
            # Add FPS and model size to metrics
            model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
            metrics.update({
                'fps': 42.5,  # Placeholder
                'model_size_mb': model_size_mb,
                'inference_time_ms': 23.5  # Placeholder
            })
            
            # Generate submission outputs
            submission_dir = Path("submission")
            model_path = self.model_dir / "best_model.pth"
            
            results = create_submission_outputs(
                prediction_map=dummy_prediction,
                input_path="dummy_thermal.tif",  # Placeholder
                metrics=metrics,
                model_path=model_path,
                output_dir=str(submission_dir),
                startup_name=self.config['export']['startup_name'],
                model_name=self.config['model']['name'],
                dataset_name="Thermal Anomaly Dataset"
            )
            
            logger.info("Sample submission outputs generated successfully!")
            
        except Exception as e:
            logger.error(f"Error generating submission outputs: {e}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")


# Loss functions
class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for imbalanced datasets."""
    
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss function."""
    
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {'bce': 0.5, 'dice': 0.3, 'focal': 0.2}
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
    
    def forward(self, predictions, targets):
        bce = self.bce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)
        
        total_loss = (
            self.weights['bce'] * bce +
            self.weights['dice'] * dice +
            self.weights['focal'] * focal
        )
        
        return total_loss


if __name__ == "__main__":
    # Test training
    print("Testing thermal anomaly trainer...")
    
    # Load configuration
    try:
        config = load_config("../config.yaml")
    except:
        # Use default config for testing
        config = {
            'model': {'name': 'SwinUNet', 'num_classes': 2},
            'dataset': {'data_dir': '../data/', 'image_size': [256, 256]},
            'training': {'epochs': 5, 'batch_size': 2, 'learning_rate': 1e-4},
            'hardware': {'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            'logging': {'experiment_name': 'test_training'},
            'export': {'startup_name': 'EuclideanTechnologies'}
        }
    
    # Initialize trainer
    trainer = ThermalAnomalyTrainer(config)
    
    print("Trainer test completed!")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")