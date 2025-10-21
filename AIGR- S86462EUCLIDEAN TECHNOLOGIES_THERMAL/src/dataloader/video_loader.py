#!/usr/bin/env python3
"""
Thermal Video Loader for real-time thermal anomaly detection.
Supports video streams and temporal fusion for ConvLSTM models.
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Iterator
from pathlib import Path
import logging
from collections import deque

logger = logging.getLogger(__name__)


class ThermalVideoLoader:
    """
    Loader for thermal video streams with temporal buffering.
    Supports real-time processing and temporal fusion.
    """
    
    def __init__(self,
                 sequence_length: int = 5,
                 target_size: Optional[Tuple[int, int]] = None,
                 normalize: bool = True,
                 temperature_range: Tuple[float, float] = (250.0, 400.0),
                 buffer_size: int = 100):
        """
        Initialize thermal video loader.
        
        Args:
            sequence_length: Number of frames for temporal models
            target_size: Target frame size (height, width)
            normalize: Whether to normalize temperature values
            temperature_range: Valid temperature range (min_temp, max_temp)
            buffer_size: Maximum number of frames to buffer
        """
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.normalize = normalize
        self.temperature_range = temperature_range
        self.buffer_size = buffer_size
        
        # Frame buffer for temporal sequences
        self.frame_buffer = deque(maxlen=buffer_size)
        self.current_sequence = deque(maxlen=sequence_length)
        
        # Video properties
        self.fps = None
        self.frame_count = None
        self.frame_width = None
        self.frame_height = None
    
    def load_video(self, video_path: Union[str, Path]) -> cv2.VideoCapture:
        """
        Load video file and extract properties.
        
        Args:
            video_path: Path to video file
            
        Returns:
            OpenCV VideoCapture object
        """
        video_path = str(video_path)
        
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Loading video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Extract video properties
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {self.frame_width}x{self.frame_height}, "
                   f"{self.fps} FPS, {self.frame_count} frames")
        
        return cap
    
    def load_camera_stream(self, camera_id: int = 0) -> cv2.VideoCapture:
        """
        Load camera stream for real-time processing.
        
        Args:
            camera_id: Camera device ID
            
        Returns:
            OpenCV VideoCapture object
        """
        logger.info(f"Loading camera stream: {camera_id}")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")
        
        # Set camera properties for thermal cameras
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Extract actual properties
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Camera properties: {self.frame_width}x{self.frame_height}, {self.fps} FPS")
        
        return cap
    
    def process_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Process a single thermal frame.
        
        Args:
            frame: Raw frame from video/camera
            
        Returns:
            Processed thermal tensor
        """
        # Convert to grayscale if needed (thermal cameras often output grayscale)
        if len(frame.shape) == 3:
            # For thermal cameras, often use only one channel or convert to grayscale
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = frame[:, :, 0]
        
        # Convert to float32
        thermal_data = frame.astype(np.float32)
        
        # Convert pixel values to temperature if needed
        # This depends on the thermal camera calibration
        thermal_data = self._pixel_to_temperature(thermal_data)
        
        # Clip to valid temperature range
        thermal_data = np.clip(thermal_data, self.temperature_range[0], self.temperature_range[1])
        
        # Normalize if requested
        if self.normalize:
            thermal_data = self._normalize_temperature(thermal_data)
        
        # Resize if requested
        if self.target_size:
            thermal_data = cv2.resize(
                thermal_data,
                (self.target_size[1], self.target_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Convert to tensor
        tensor = torch.from_numpy(thermal_data).float()
        
        # Add channel dimension
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def get_frame_sequence(self, cap: cv2.VideoCapture) -> Iterator[torch.Tensor]:
        """
        Generate frame sequences for temporal models.
        
        Args:
            cap: OpenCV VideoCapture object
            
        Yields:
            Tensor sequences of shape (sequence_length, channels, height, width)
        """
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Add to buffer
            self.frame_buffer.append(processed_frame)
            self.current_sequence.append(processed_frame)
            
            # Yield sequence when we have enough frames
            if len(self.current_sequence) == self.sequence_length:
                sequence = torch.stack(list(self.current_sequence))
                yield sequence
    
    def get_single_frames(self, cap: cv2.VideoCapture) -> Iterator[torch.Tensor]:
        """
        Generate single frames for non-temporal models.
        
        Args:
            cap: OpenCV VideoCapture object
            
        Yields:
            Single frame tensors
        """
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            yield processed_frame
    
    def _pixel_to_temperature(self, pixel_values: np.ndarray) -> np.ndarray:
        """
        Convert pixel values to temperature.
        This is a simplified conversion - in practice, this would depend on
        the specific thermal camera calibration.
        
        Args:
            pixel_values: Raw pixel values
            
        Returns:
            Temperature values in Kelvin
        """
        # Simple linear mapping from pixel values to temperature
        # Adjust these parameters based on your thermal camera specifications
        min_pixel = 0
        max_pixel = 255
        min_temp, max_temp = self.temperature_range
        
        # Linear interpolation
        temperature = min_temp + (pixel_values - min_pixel) * (max_temp - min_temp) / (max_pixel - min_pixel)
        
        return temperature
    
    def _normalize_temperature(self, data: np.ndarray) -> np.ndarray:
        """Normalize temperature data to [0, 1] range."""
        min_temp, max_temp = self.temperature_range
        normalized = (data - min_temp) / (max_temp - min_temp)
        return np.clip(normalized, 0.0, 1.0)
    
    def reset_buffer(self):
        """Reset frame buffers."""
        self.frame_buffer.clear()
        self.current_sequence.clear()
    
    def get_video_info(self) -> Dict:
        """Get video information."""
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'sequence_length': self.sequence_length,
            'target_size': self.target_size
        }


class ThermalVideoDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for thermal video sequences.
    """
    
    def __init__(self,
                 video_paths: List[Union[str, Path]],
                 annotations: Optional[List[Dict]] = None,
                 sequence_length: int = 5,
                 target_size: Optional[Tuple[int, int]] = None,
                 normalize: bool = True,
                 stride: int = 1):
        """
        Initialize thermal video dataset.
        
        Args:
            video_paths: List of video file paths
            annotations: List of annotation dictionaries (optional)
            sequence_length: Number of frames per sequence
            target_size: Target frame size
            normalize: Whether to normalize frames
            stride: Frame stride for sequence extraction
        """
        self.video_paths = [Path(p) for p in video_paths]
        self.annotations = annotations or []
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.normalize = normalize
        self.stride = stride
        
        # Initialize video loader
        self.loader = ThermalVideoLoader(
            sequence_length=sequence_length,
            target_size=target_size,
            normalize=normalize
        )
        
        # Extract all sequences
        self._extract_sequences()
    
    def _extract_sequences(self):
        """Extract all frame sequences from videos."""
        self.sequences = []
        self.sequence_annotations = []
        
        for i, video_path in enumerate(self.video_paths):
            logger.info(f"Extracting sequences from {video_path}")
            
            cap = self.loader.load_video(video_path)
            
            # Get annotation for this video if available
            video_annotation = self.annotations[i] if i < len(self.annotations) else {}
            
            sequence_count = 0
            for sequence in self.loader.get_frame_sequence(cap):
                self.sequences.append(sequence)
                self.sequence_annotations.append(video_annotation)
                sequence_count += 1
            
            cap.release()
            logger.info(f"Extracted {sequence_count} sequences from {video_path}")
        
        logger.info(f"Total sequences: {len(self.sequences)}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        annotation = self.sequence_annotations[idx]
        
        item = {
            'sequence': sequence,
            'video_idx': idx
        }
        
        # Add annotation data if available
        if annotation:
            for key, value in annotation.items():
                if isinstance(value, (int, float, list, np.ndarray)):
                    if isinstance(value, np.ndarray):
                        item[key] = torch.from_numpy(value)
                    else:
                        item[key] = torch.tensor(value)
        
        return item


def create_thermal_video_dataloader(video_paths: List[Union[str, Path]],
                                  batch_size: int = 4,
                                  num_workers: int = 4,
                                  shuffle: bool = True,
                                  **dataset_kwargs) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for thermal video data.
    
    Args:
        video_paths: List of video file paths
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        **dataset_kwargs: Additional arguments for ThermalVideoDataset
        
    Returns:
        PyTorch DataLoader
    """
    dataset = ThermalVideoDataset(video_paths, **dataset_kwargs)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Test video loading
    loader = ThermalVideoLoader(
        sequence_length=5,
        target_size=(512, 512),
        normalize=True
    )
    
    # Test with camera (if available)
    try:
        cap = loader.load_camera_stream(0)
        print("Camera loaded successfully")
        
        # Get a few frames
        for i, frame in enumerate(loader.get_single_frames(cap)):
            print(f"Frame {i}: {frame.shape}")
            if i >= 5:
                break
        
        cap.release()
    except Exception as e:
        print(f"Camera test failed: {e}")
    
    print("Video loader test completed")