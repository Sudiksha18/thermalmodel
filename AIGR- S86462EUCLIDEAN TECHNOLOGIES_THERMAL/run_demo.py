#!/usr/bin/env python3
"""
Thermal Anomaly Detection Demo Runner
Runs the organized demos from the root directory
"""
import os
import sys
import subprocess
from pathlib import Path

def run_quick_demo():
    """Run the quick demo from the demos folder"""
    print("Running Quick Demo...")
    print("=" * 50)
    
    # Change to demos directory and run
    demos_dir = Path("demos")
    original_dir = os.getcwd()
    
    try:
        os.chdir(demos_dir)
        result = subprocess.run([sys.executable, "quick_demo.py"], 
                              capture_output=False, text=True)
        return result.returncode == 0
    finally:
        os.chdir(original_dir)

def run_main():
    """Run the main application"""
    print("Running Main Application...")
    print("=" * 50)
    
    # Run main.py with inference mode
    result = subprocess.run([sys.executable, "main.py", "--mode", "inference", 
                           "--config", "config/config.yaml"], 
                          capture_output=False, text=True)
    return result.returncode == 0

def main():
    print("EUCLIDEAN TECHNOLOGIES - THERMAL ANOMALY DETECTION")
    print("=" * 60)
    print("Available options:")
    print("1. Quick Demo (Recommended)")
    print("2. Main Application")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            success = run_quick_demo()
            if success:
                print("\n✓ Quick demo completed successfully!")
            else:
                print("\n✗ Quick demo failed!")
            break
        elif choice == "2":
            success = run_main()
            if success:
                print("\n✓ Main application completed successfully!")
            else:
                print("\n✗ Main application failed!")
            break
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()