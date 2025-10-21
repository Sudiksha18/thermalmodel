#!/usr/bin/env python3
"""
Dataset utilities and PyTorch Lightning DataModule for thermal anomaly detection.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from .he5_loader import HE5ThermalLoader, load_thermal_file
from .video_loader import ThermalVideoLoader
from .augmentations import ThermalAugmentations, create_thermal_augmentations

import logging
logger = logging.getLogger(__name__)


class ThermalDataset(Dataset):
    """
    PyTorch Dataset for thermal anomaly detection.
    Supports both images and videos, with optional annotations.
    """
    
    def __init__(self,
                 data_paths: List[Union[str, Path]],
                 annotations: Optional[List[Dict]] = None,
                 image_size: Tuple[int, int] = (512, 512),
                 normalize: bool = True,
                 augmentations: Optional[ThermalAugmentations] = None,
                 mode: str = "train",
                 temperature_range: Tuple[float, float] = (250.0, 400.0),
                 return_metadata: bool = False):
        """
        Initialize thermal dataset.
        
        Args:
            data_paths: List of thermal file paths
            annotations: List of annotation dictionaries
            image_size: Target image size (height, width)
            normalize: Whether to normalize thermal data
            augmentations: Augmentation transforms
            mode: Dataset mode ("train", "val", "test")
            temperature_range: Valid temperature range
            return_metadata: Whether to return metadata
        """
        self.data_paths = [Path(p) for p in data_paths]
        self.annotations = annotations or [{}] * len(data_paths)
        self.image_size = image_size
        self.normalize = normalize
        self.augmentations = augmentations
        self.mode = mode
        self.temperature_range = temperature_range
        self.return_metadata = return_metadata
        
        # Initialize thermal loader
        self.thermal_loader = HE5ThermalLoader(
            normalize=normalize,
            target_size=image_size,
            temperature_range=temperature_range
        )
        
        # Validate data
        self._validate_data()
        
        logger.info(f"Initialized ThermalDataset with {len(self.data_paths)} samples in {mode} mode")
    
    def _validate_data(self):
        """Validate that all data files exist."""
        valid_paths = []
        valid_annotations = []
        
        for i, path in enumerate(self.data_paths):
            if path.exists():
                valid_paths.append(path)
                valid_annotations.append(self.annotations[i] if i < len(self.annotations) else {})
            else:
                logger.warning(f"File not found: {path}")
        
        self.data_paths = valid_paths
        self.annotations = valid_annotations
        
        if len(self.data_paths) == 0:
            raise ValueError("No valid data files found")
    
    def __len__(self) -> int:
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing thermal data and annotations
        """
        # Load thermal data
        file_path = self.data_paths[idx]
        annotation = self.annotations[idx]
        
        try:
            # Load thermal file
            thermal_data = load_thermal_file(
                file_path,
                normalize=self.normalize,
                target_size=self.image_size,
                temperature_range=self.temperature_range
            )
            
            thermal_tensor = thermal_data['thermal_data']
            metadata = thermal_data['metadata']
            
            # Convert tensor to numpy for augmentations
            if isinstance(thermal_tensor, torch.Tensor):
                thermal_image = thermal_tensor.squeeze().numpy()
            else:
                thermal_image = thermal_tensor
            
            # Load mask if available
            mask = None
            if 'mask_path' in annotation:
                mask = self._load_mask(annotation['mask_path'])
            elif 'anomaly_mask' in annotation:
                mask = np.array(annotation['anomaly_mask'])
            
            # Apply augmentations
            if self.augmentations is not None:
                augmented = self.augmentations(thermal_image, mask, mode=self.mode)
                thermal_tensor = augmented['image']
                if mask is not None:
                    mask = augmented['mask']
            else:
                # Convert to tensor
                thermal_tensor = torch.from_numpy(thermal_image).float()
                if thermal_tensor.dim() == 2:
                    thermal_tensor = thermal_tensor.unsqueeze(0)
                
                if mask is not None:
                    mask = torch.from_numpy(mask).long()
            
            # Prepare return dictionary
            item = {
                'image': thermal_tensor,
                'idx': torch.tensor(idx),
                'file_path': str(file_path)
            }
            
            # Add mask if available
            if mask is not None:
                item['mask'] = mask
            
            # Add labels if available
            if 'label' in annotation:
                item['label'] = torch.tensor(annotation['label'], dtype=torch.long)
            
            # Add bounding boxes if available
            if 'bbox' in annotation:
                item['bbox'] = torch.tensor(annotation['bbox'], dtype=torch.float32)
            
            # Add metadata if requested
            if self.return_metadata:
                item['metadata'] = metadata
            
            return item
            
        except Exception as e:
            logger.error(f"Error loading sample {idx} ({file_path}): {str(e)}")
            # Return dummy data to avoid breaking training
            return self._get_dummy_sample(idx)
    
    def _load_mask(self, mask_path: Union[str, Path]) -> np.ndarray:
        """Load segmentation mask."""
        mask_path = Path(mask_path)
        
        if not mask_path.exists():
            logger.warning(f"Mask file not found: {mask_path}")
            return np.zeros(self.image_size, dtype=np.uint8)
        
        try:
            if mask_path.suffix.lower() in ['.tif', '.tiff']:
                import rasterio
                with rasterio.open(mask_path) as src:
                    mask = src.read(1).astype(np.uint8)
            else:
                import cv2
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Resize if needed
            if mask.shape != self.image_size:
                import cv2
                mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
            
            return mask
            
        except Exception as e:
            logger.error(f"Error loading mask {mask_path}: {str(e)}")
            return np.zeros(self.image_size, dtype=np.uint8)
    
    def _get_dummy_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dummy sample for error cases."""
        return {
            'image': torch.zeros(1, *self.image_size),
            'mask': torch.zeros(self.image_size, dtype=torch.long),
            'idx': torch.tensor(idx),
            'file_path': "dummy"
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        if not hasattr(self, '_class_weights'):
            logger.info("Calculating class weights...")
            
            label_counts = {}
            for annotation in self.annotations:
                if 'label' in annotation:
                    label = annotation['label']
                    label_counts[label] = label_counts.get(label, 0) + 1
            
            if label_counts:
                total_samples = sum(label_counts.values())
                num_classes = len(label_counts)
                
                # Calculate inverse frequency weights
                weights = []
                for i in range(num_classes):
                    count = label_counts.get(i, 1)
                    weight = total_samples / (num_classes * count)
                    weights.append(weight)
                
                self._class_weights = torch.tensor(weights, dtype=torch.float32)
            else:
                # Default weights
                self._class_weights = torch.tensor([1.0, 1.0])
        
        return self._class_weights


class ThermalDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for thermal anomaly detection.
    """
    
    def __init__(self,
                 config: Dict,
                 data_dir: Optional[str] = None,
                 annotations_file: Optional[str] = None):
        """
        Initialize thermal data module.
        
        Args:
            config: Configuration dictionary
            data_dir: Data directory path
            annotations_file: Annotations file path
        """
        super().__init__()
        self.config = config
        self.data_dir = Path(data_dir) if data_dir else Path(config['dataset']['data_dir'])
        self.annotations_file = annotations_file
        
        # Dataset configuration
        self.dataset_config = config['dataset']
        self.training_config = config['training']
        
        # Data splits
        self.train_split = self.dataset_config.get('train_split', 0.7)
        self.val_split = self.dataset_config.get('val_split', 0.15)
        self.test_split = self.dataset_config.get('test_split', 0.15)
        
        # Image settings
        self.image_size = tuple(self.dataset_config['image_size'])
        self.normalize = self.dataset_config.get('normalize', True)
        
        # Training settings
        self.batch_size = self.training_config['batch_size']
        self.num_workers = config['hardware'].get('num_workers', 4)
        
        # Initialize augmentations
        self.augmentations = create_thermal_augmentations(config)
        
        # Data containers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """Prepare data (download, extract, etc.)."""
        # Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Find thermal files
        thermal_files = self._find_thermal_files()
        
        if len(thermal_files) == 0:
            raise ValueError(f"No thermal files found in {self.data_dir}")
        
        logger.info(f"Found {len(thermal_files)} thermal files")
        
        # Load annotations if available
        annotations = self._load_annotations()
        
        # Create data splits
        self._create_data_splits(thermal_files, annotations)
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages."""
        if stage == "fit" or stage is None:
            # Training and validation datasets
            if not hasattr(self, 'train_files') or not hasattr(self, 'val_files'):
                self.prepare_data()
            
            self.train_dataset = ThermalDataset(
                data_paths=self.train_files,
                annotations=self.train_annotations,
                image_size=self.image_size,
                normalize=self.normalize,
                augmentations=self.augmentations,
                mode="train"
            )
            
            self.val_dataset = ThermalDataset(
                data_paths=self.val_files,
                annotations=self.val_annotations,
                image_size=self.image_size,
                normalize=self.normalize,
                augmentations=self.augmentations,
                mode="val"
            )
        
        if stage == "test" or stage is None:
            # Test dataset
            if not hasattr(self, 'test_files'):
                self.prepare_data()
            
            self.test_dataset = ThermalDataset(
                data_paths=self.test_files,
                annotations=self.test_annotations,
                image_size=self.image_size,
                normalize=self.normalize,
                augmentations=None,  # No augmentation for test
                mode="test"
            )
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def _find_thermal_files(self) -> List[Path]:
        """Find all thermal files in data directory."""
        thermal_files = []
        
        supported_formats = self.dataset_config.get('formats', ['.tif', '.he5', '.png', '.jpg'])
        
        for format_ext in supported_formats:
            files = list(self.data_dir.rglob(f"*{format_ext}"))
            thermal_files.extend(files)
        
        return sorted(thermal_files)
    
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from file."""
        if not self.annotations_file or not Path(self.annotations_file).exists():
            logger.info("No annotations file provided or found")
            return []
        
        try:
            if self.annotations_file.endswith('.json'):
                with open(self.annotations_file, 'r') as f:
                    annotations = json.load(f)
            elif self.annotations_file.endswith('.csv'):
                df = pd.read_csv(self.annotations_file)
                annotations = df.to_dict('records')
            else:
                logger.warning(f"Unsupported annotations format: {self.annotations_file}")
                return []
            
            logger.info(f"Loaded {len(annotations)} annotations")
            return annotations
            
        except Exception as e:
            logger.error(f"Error loading annotations: {str(e)}")
            return []
    
    def _create_data_splits(self, thermal_files: List[Path], annotations: List[Dict]):
        """Create train/val/test splits."""
        # Check if splits already exist
        splits_dir = self.data_dir / "splits"
        
        if splits_dir.exists():
            # Load existing splits
            self._load_existing_splits(splits_dir)
        else:
            # Create new splits
            self._create_new_splits(thermal_files, annotations, splits_dir)
    
    def _load_existing_splits(self, splits_dir: Path):
        """Load existing data splits."""
        try:
            with open(splits_dir / "train.json", 'r') as f:
                train_data = json.load(f)
            with open(splits_dir / "val.json", 'r') as f:
                val_data = json.load(f)
            with open(splits_dir / "test.json", 'r') as f:
                test_data = json.load(f)
            
            self.train_files = [Path(p) for p in train_data['files']]
            self.train_annotations = train_data['annotations']
            
            self.val_files = [Path(p) for p in val_data['files']]
            self.val_annotations = val_data['annotations']
            
            self.test_files = [Path(p) for p in test_data['files']]
            self.test_annotations = test_data['annotations']
            
            logger.info(f"Loaded existing splits: {len(self.train_files)} train, "
                       f"{len(self.val_files)} val, {len(self.test_files)} test")
            
        except Exception as e:
            logger.error(f"Error loading existing splits: {str(e)}")
            raise
    
    def _create_new_splits(self, thermal_files: List[Path], annotations: List[Dict], splits_dir: Path):
        """Create new data splits."""
        # Create splits directory
        splits_dir.mkdir(exist_ok=True)
        
        # Match annotations to files if available
        file_annotations = {}
        for annotation in annotations:
            if 'file_path' in annotation:
                file_path = Path(annotation['file_path'])
                file_annotations[file_path.name] = annotation
        
        # Create matched annotations list
        matched_annotations = []
        for file_path in thermal_files:
            if file_path.name in file_annotations:
                matched_annotations.append(file_annotations[file_path.name])
            else:
                matched_annotations.append({})
        
        # Create train/val/test splits
        indices = list(range(len(thermal_files)))
        
        # First split: train + val vs test
        train_val_indices, test_indices = train_test_split(
            indices, test_size=self.test_split, random_state=42
        )
        
        # Second split: train vs val
        val_ratio = self.val_split / (self.train_split + self.val_split)
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_ratio, random_state=42
        )
        
        # Create file lists and annotations
        self.train_files = [thermal_files[i] for i in train_indices]
        self.train_annotations = [matched_annotations[i] for i in train_indices]
        
        self.val_files = [thermal_files[i] for i in val_indices]
        self.val_annotations = [matched_annotations[i] for i in val_indices]
        
        self.test_files = [thermal_files[i] for i in test_indices]
        self.test_annotations = [matched_annotations[i] for i in test_indices]
        
        # Save splits
        train_data = {
            'files': [str(p) for p in self.train_files],
            'annotations': self.train_annotations
        }
        val_data = {
            'files': [str(p) for p in self.val_files],
            'annotations': self.val_annotations
        }
        test_data = {
            'files': [str(p) for p in self.test_files],
            'annotations': self.test_annotations
        }
        
        with open(splits_dir / "train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        with open(splits_dir / "val.json", 'w') as f:
            json.dump(val_data, f, indent=2)
        with open(splits_dir / "test.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        
        logger.info(f"Created new splits: {len(self.train_files)} train, "
                   f"{len(self.val_files)} val, {len(self.test_files)} test")


if __name__ == "__main__":
    # Test dataset
    from pathlib import Path
    
    # Create dummy config
    config = {
        'dataset': {
            'data_dir': 'data/',
            'image_size': [512, 512],
            'normalize': True,
            'formats': ['.tif', '.he5'],
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15
        },
        'training': {
            'batch_size': 4
        },
        'hardware': {
            'num_workers': 2
        }
    }
    
    # Test data module
    try:
        data_module = ThermalDataModule(config)
        data_module.prepare_data()
        data_module.setup("fit")
        
        # Test train dataloader
        train_loader = data_module.train_dataloader()
        for batch in train_loader:
            print(f"Batch shape: {batch['image'].shape}")
            break
        
        print("Dataset test completed successfully!")
        
    except Exception as e:
        print(f"Dataset test failed: {e}")
        print("This is expected if no data files are available")