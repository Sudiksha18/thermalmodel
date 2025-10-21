#!/usr/bin/env python3
"""
Thermal-safe augmentations for thermal anomaly detection.
Preserves temperature relationships while providing data augmentation.
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union, Callable
import random
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

import logging
logger = logging.getLogger(__name__)


class ThermalAugmentations:
    """
    Thermal-safe augmentations that preserve temperature relationships.
    """
    
    def __init__(self,
                 geometric_prob: float = 0.5,
                 noise_prob: float = 0.3,
                 blur_prob: float = 0.2,
                 contrast_prob: float = 0.4,
                 preserve_temperature: bool = True):
        """
        Initialize thermal augmentations.
        
        Args:
            geometric_prob: Probability of geometric transformations
            noise_prob: Probability of noise augmentation
            blur_prob: Probability of blur augmentation  
            contrast_prob: Probability of contrast adjustment
            preserve_temperature: Whether to preserve temperature relationships
        """
        self.geometric_prob = geometric_prob
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob
        self.contrast_prob = contrast_prob
        self.preserve_temperature = preserve_temperature
        
        # Initialize augmentation pipelines
        self.train_transforms = self._create_train_transforms()
        self.val_transforms = self._create_val_transforms()
        self.test_transforms = self._create_test_transforms()
    
    def _create_train_transforms(self) -> A.Compose:
        """Create training augmentations."""
        augmentations = []
        
        # Geometric transformations (safe for thermal data)
        if self.geometric_prob > 0:
            augmentations.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.3
                ),
                A.ElasticTransform(
                    alpha=50,
                    sigma=5,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.2
                )
            ])
        
        # Noise augmentations (thermal-safe)
        if self.noise_prob > 0:
            augmentations.extend([
                A.GaussNoise(var_limit=(0.001, 0.01), p=self.noise_prob),
                A.ISONoise(color_shift=(0.01, 0.05), p=self.noise_prob * 0.5)
            ])
        
        # Blur augmentations (simulate atmospheric effects)
        if self.blur_prob > 0:
            augmentations.extend([
                A.GaussianBlur(blur_limit=(3, 5), p=self.blur_prob),
                A.MotionBlur(blur_limit=3, p=self.blur_prob * 0.5)
            ])
        
        # Contrast adjustments (careful with thermal data)
        if self.contrast_prob > 0 and not self.preserve_temperature:
            augmentations.extend([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=self.contrast_prob * 0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=self.contrast_prob
                )
            ])
        
        # Normalize and convert to tensor
        augmentations.append(A.Normalize(mean=0.0, std=1.0))
        augmentations.append(ToTensorV2())
        
        return A.Compose(augmentations)
    
    def _create_val_transforms(self) -> A.Compose:
        """Create validation transforms (minimal augmentation)."""
        return A.Compose([
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2()
        ])
    
    def _create_test_transforms(self) -> A.Compose:
        """Create test transforms (no augmentation)."""
        return A.Compose([
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2()
        ])
    
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None, mode: str = "train") -> Dict:
        """
        Apply augmentations to thermal image and mask.
        
        Args:
            image: Thermal image array
            mask: Optional segmentation mask
            mode: Augmentation mode ("train", "val", "test")
            
        Returns:
            Dictionary with augmented image and mask
        """
        # Select appropriate transforms
        if mode == "train":
            transform = self.train_transforms
        elif mode == "val":
            transform = self.val_transforms
        else:
            transform = self.test_transforms
        
        # Ensure image is float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Apply transforms
        if mask is not None:
            # Ensure mask is uint8
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            
            augmented = transform(image=image, mask=mask)
            return {
                'image': augmented['image'],
                'mask': augmented['mask']
            }
        else:
            augmented = transform(image=image)
            return {
                'image': augmented['image']
            }


class ThermalTemporalAugmentations:
    """
    Augmentations for temporal thermal sequences (videos).
    """
    
    def __init__(self,
                 sequence_length: int = 5,
                 temporal_shift_prob: float = 0.3,
                 frame_drop_prob: float = 0.2,
                 temporal_noise_prob: float = 0.3):
        """
        Initialize temporal augmentations.
        
        Args:
            sequence_length: Length of temporal sequences
            temporal_shift_prob: Probability of temporal shifting
            frame_drop_prob: Probability of frame dropping
            temporal_noise_prob: Probability of temporal noise
        """
        self.sequence_length = sequence_length
        self.temporal_shift_prob = temporal_shift_prob
        self.frame_drop_prob = frame_drop_prob
        self.temporal_noise_prob = temporal_noise_prob
        
        # Spatial augmentations
        self.spatial_augmentations = ThermalAugmentations()
    
    def __call__(self, sequence: torch.Tensor, masks: Optional[torch.Tensor] = None, mode: str = "train") -> Dict:
        """
        Apply temporal augmentations to thermal sequence.
        
        Args:
            sequence: Thermal sequence tensor (T, C, H, W)
            masks: Optional mask sequence (T, H, W)
            mode: Augmentation mode
            
        Returns:
            Dictionary with augmented sequence and masks
        """
        if mode != "train":
            # No temporal augmentation for val/test
            return {'sequence': sequence, 'masks': masks}
        
        augmented_sequence = sequence.clone()
        augmented_masks = masks.clone() if masks is not None else None
        
        # Temporal shifting
        if random.random() < self.temporal_shift_prob:
            augmented_sequence, augmented_masks = self._temporal_shift(
                augmented_sequence, augmented_masks
            )
        
        # Frame dropping and interpolation
        if random.random() < self.frame_drop_prob:
            augmented_sequence, augmented_masks = self._frame_drop_interpolate(
                augmented_sequence, augmented_masks
            )
        
        # Temporal noise
        if random.random() < self.temporal_noise_prob:
            augmented_sequence = self._add_temporal_noise(augmented_sequence)
        
        return {
            'sequence': augmented_sequence,
            'masks': augmented_masks
        }
    
    def _temporal_shift(self, sequence: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply random temporal shift."""
        max_shift = min(2, sequence.size(0) // 4)
        shift = random.randint(-max_shift, max_shift)
        
        if shift > 0:
            # Shift forward (repeat first frames)
            sequence = torch.cat([sequence[:1].repeat(shift, 1, 1, 1), sequence[:-shift]], dim=0)
            if masks is not None:
                masks = torch.cat([masks[:1].repeat(shift, 1, 1), masks[:-shift]], dim=0)
        elif shift < 0:
            # Shift backward (repeat last frames)
            sequence = torch.cat([sequence[-shift:], sequence[-1:].repeat(-shift, 1, 1, 1)], dim=0)
            if masks is not None:
                masks = torch.cat([masks[-shift:], masks[-1:].repeat(-shift, 1, 1)], dim=0)
        
        return sequence, masks
    
    def _frame_drop_interpolate(self, sequence: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Drop random frames and interpolate."""
        seq_len = sequence.size(0)
        
        # Don't drop too many frames
        max_drops = min(2, seq_len // 3)
        num_drops = random.randint(1, max_drops)
        
        # Select frames to drop (not first or last)
        drop_indices = random.sample(range(1, seq_len - 1), num_drops)
        
        for idx in sorted(drop_indices, reverse=True):
            # Linear interpolation between adjacent frames
            prev_frame = sequence[idx - 1]
            next_frame = sequence[idx + 1] if idx + 1 < seq_len else sequence[idx - 1]
            interpolated = (prev_frame + next_frame) / 2
            
            sequence[idx] = interpolated
            
            if masks is not None:
                # For masks, use nearest neighbor
                masks[idx] = masks[idx - 1]
        
        return sequence, masks
    
    def _add_temporal_noise(self, sequence: torch.Tensor) -> torch.Tensor:
        """Add temporal noise to sequence."""
        # Add small random variations across time
        temporal_noise = torch.randn_like(sequence) * 0.01
        
        # Apply temporal smoothing to noise
        for t in range(1, sequence.size(0)):
            temporal_noise[t] = 0.7 * temporal_noise[t] + 0.3 * temporal_noise[t-1]
        
        return sequence + temporal_noise


class ThermalTestTimeAugmentation:
    """
    Test-Time Augmentation (TTA) for thermal anomaly detection.
    """
    
    def __init__(self, 
                 tta_transforms: Optional[List[str]] = None,
                 merge_mode: str = "mean"):
        """
        Initialize TTA.
        
        Args:
            tta_transforms: List of TTA transform names
            merge_mode: How to merge predictions ("mean", "max", "vote")
        """
        self.tta_transforms = tta_transforms or ["original", "hflip", "vflip", "rotate90"]
        self.merge_mode = merge_mode
    
    def __call__(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate TTA versions of image.
        
        Args:
            image: Input thermal image tensor
            
        Returns:
            List of augmented image tensors
        """
        augmented_images = []
        
        for transform_name in self.tta_transforms:
            if transform_name == "original":
                augmented_images.append(image.clone())
            elif transform_name == "hflip":
                augmented_images.append(torch.flip(image, dims=[-1]))
            elif transform_name == "vflip":
                augmented_images.append(torch.flip(image, dims=[-2]))
            elif transform_name == "rotate90":
                augmented_images.append(torch.rot90(image, k=1, dims=[-2, -1]))
            elif transform_name == "rotate180":
                augmented_images.append(torch.rot90(image, k=2, dims=[-2, -1]))
            elif transform_name == "rotate270":
                augmented_images.append(torch.rot90(image, k=3, dims=[-2, -1]))
        
        return augmented_images
    
    def merge_predictions(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Merge TTA predictions.
        
        Args:
            predictions: List of prediction tensors
            
        Returns:
            Merged prediction tensor
        """
        if self.merge_mode == "mean":
            return torch.stack(predictions).mean(dim=0)
        elif self.merge_mode == "max":
            return torch.stack(predictions).max(dim=0)[0]
        elif self.merge_mode == "vote":
            # Majority vote for binary predictions
            binary_preds = torch.stack(predictions) > 0.5
            return binary_preds.float().mean(dim=0)
        else:
            raise ValueError(f"Unknown merge mode: {self.merge_mode}")


def create_thermal_augmentations(config: Dict) -> ThermalAugmentations:
    """
    Create thermal augmentations from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ThermalAugmentations instance
    """
    dataset_config = config.get('dataset', {})
    
    return ThermalAugmentations(
        geometric_prob=dataset_config.get('geometric_prob', 0.5),
        noise_prob=dataset_config.get('noise_prob', 0.3),
        blur_prob=dataset_config.get('blur_prob', 0.2),
        contrast_prob=dataset_config.get('contrast_prob', 0.4),
        preserve_temperature=dataset_config.get('preserve_temperature', True)
    )


if __name__ == "__main__":
    # Test augmentations
    augmentations = ThermalAugmentations()
    
    # Create dummy thermal image
    thermal_image = np.random.rand(512, 512).astype(np.float32) * 100 + 250  # Thermal range 250-350K
    thermal_mask = np.random.randint(0, 2, (512, 512)).astype(np.uint8)
    
    # Test training augmentations
    result = augmentations(thermal_image, thermal_mask, mode="train")
    print(f"Augmented image shape: {result['image'].shape}")
    print(f"Augmented mask shape: {result['mask'].shape}")
    
    # Test temporal augmentations
    temporal_aug = ThermalTemporalAugmentations(sequence_length=5)
    sequence = torch.randn(5, 1, 512, 512)
    masks = torch.randint(0, 2, (5, 512, 512))
    
    temporal_result = temporal_aug(sequence, masks, mode="train")
    print(f"Temporal sequence shape: {temporal_result['sequence'].shape}")
    print(f"Temporal masks shape: {temporal_result['masks'].shape}")
    
    print("Augmentation tests completed!")