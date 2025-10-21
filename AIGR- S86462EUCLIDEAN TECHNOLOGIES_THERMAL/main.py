#!/usr/bin/env python3
"""
Main entry point for Euclidean Technologies Thermal Anomaly Detection System.
Supports training, validation, inference, and batch processing modes.
"""

import argparse
import os
import sys
import torch
import yaml
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config_parser import load_config
from src.utils.logging_utils import setup_logger
from src.train.train import ThermalAnomalyTrainer
from src.inference.export_submission import SubmissionGenerator
from src.inference.thermal_inference import ThermalAnomalyInference


def main():
    parser = argparse.ArgumentParser(description="Thermal Anomaly Detection System")
    parser.add_argument("--mode", type=str, required=True, 
                       choices=["train", "test", "inference", "batch_inference", "video_inference"],
                       help="Mode of operation")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--input_path", type=str, default=None,
                       help="Input file path for inference")
    parser.add_argument("--input_dir", type=str, default=None,
                       help="Input directory for batch inference")
    parser.add_argument("--output_dir", type=str, default="submission/",
                       help="Output directory for results")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to trained model for inference")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size override")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Anomaly threshold for inference")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        config['inference']['batch_size'] = args.batch_size
    if args.device:
        config['hardware']['device'] = args.device
    if args.threshold:
        config['inference']['threshold'] = args.threshold
    
    # Setup logging
    logger = setup_logger(config['logging'], debug=args.debug)
    logger.info(f"Starting Thermal Anomaly Detection System in {args.mode} mode")
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU")
        config['hardware']['device'] = "cpu"
    
    # Run based on mode
    try:
        if args.mode == "train":
            run_training(config, args, logger)
        elif args.mode == "test":
            run_testing(config, args, logger)
        elif args.mode == "inference":
            run_inference(config, args, logger)
        elif args.mode == "batch_inference":
            run_batch_inference(config, args, logger)
        elif args.mode == "video_inference":
            run_video_inference(config, args, logger)
    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {str(e)}")
        raise


def run_training(config, args, logger):
    """Run model training."""
    logger.info("Starting model training...")
    
    trainer = ThermalAnomalyTrainer(config)
    
    if args.resume:
        logger.info(f"Resuming training from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train the model
    trainer.train()
    
    logger.info("Training completed successfully!")


def run_testing(config, args, logger):
    """Run model testing/evaluation."""
    logger.info("Starting model testing...")
    
    if not args.model_path:
        args.model_path = config['inference']['model_path']
    
    trainer = ThermalAnomalyTrainer(config)
    trainer.load_model(args.model_path)
    
    # Run evaluation
    results = trainer.evaluate_test_set()
    
    logger.info(f"Testing completed! Results: {results}")


def run_inference(config, args, logger):
    """Run inference on a single image."""
    if not args.input_path:
        raise ValueError("Input path required for inference mode")
    
    if not args.model_path:
        args.model_path = config['inference']['model_path']
    
    logger.info(f"Running inference on {args.input_path}")
    
    inference = ThermalAnomalyInference(config, args.model_path)
    results = inference.predict_image(
        args.input_path,
        output_dir=args.output_dir,
        save_outputs=True
    )
    
    logger.info(f"Inference completed! Output saved to {args.output_dir}")
    return results


def run_batch_inference(config, args, logger):
    """Run batch inference on multiple images."""
    if not args.input_dir:
        raise ValueError("Input directory required for batch inference mode")
    
    if not args.model_path:
        args.model_path = config['inference']['model_path']
    
    logger.info(f"Running batch inference on {args.input_dir}")
    
    inference = ThermalAnomalyInference(config, args.model_path)
    results = inference.predict_batch(
        args.input_dir,
        output_dir=args.output_dir,
        save_outputs=True
    )
    
    logger.info(f"Batch inference completed! {len(results)} images processed")
    return results


def run_video_inference(config, args, logger):
    """Run inference on video stream."""
    if not args.input_path:
        raise ValueError("Video path required for video inference mode")
    
    if not args.model_path:
        args.model_path = config['inference']['model_path']
    
    logger.info(f"Running video inference on {args.input_path}")
    
    video_inference = ThermalVideoInference(config, args.model_path)
    results = video_inference.process_video(
        args.input_path,
        output_dir=args.output_dir,
        save_outputs=True
    )
    
    logger.info(f"Video inference completed!")
    return results


if __name__ == "__main__":
    main()