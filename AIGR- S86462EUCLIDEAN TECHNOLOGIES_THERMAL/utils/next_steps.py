#!/usr/bin/env python3
"""
Next Steps Menu for Thermal Anomaly Detection System
"""

import os
from pathlib import Path

def show_menu():
    print("\n" + "=" * 70)
    print("THERMAL ANOMALY DETECTION SYSTEM - NEXT STEPS")
    print("=" * 70)
    
    print("\n📁 CURRENT STATUS:")
    print("✅ TIF file converted to MAT format")
    print("✅ Quick demo completed successfully")
    print("✅ Submission files generated")
    
    print("\n📋 AVAILABLE NEXT STEPS:")
    print("\n1. 🔍 VIEW OUTPUTS")
    print("   - Open generated PNG visualization")
    print("   - Check Excel metrics file")
    print("   - Review README documentation")
    
    print("\n2. 🚀 RUN FULL TRAINING")
    print("   - Train actual deep learning model")
    print("   - Generate real predictions")
    print("   - Evaluate on test data")
    
    print("\n3. 📊 BATCH PROCESSING")
    print("   - Process multiple thermal files")
    print("   - Generate predictions for entire dataset")
    print("   - Create comparative analysis")
    
    print("\n4. 🎥 VIDEO PROCESSING")
    print("   - Process thermal video sequences")
    print("   - Temporal anomaly detection")
    print("   - Generate video outputs")
    
    print("\n5. 📈 ADVANCED ANALYSIS")
    print("   - Statistical analysis of results")
    print("   - Performance benchmarking")
    print("   - Model comparison")
    
    print("\n6. 🔧 CUSTOMIZATION")
    print("   - Modify detection parameters")
    print("   - Adjust visualization settings")
    print("   - Custom threshold tuning")
    
    print("\n7. 📤 EXPORT & SHARE")
    print("   - Package submission files")
    print("   - Generate reports")
    print("   - Create presentation materials")
    
    print("\n" + "=" * 70)
    
    # Show current files
    print("\n📂 GENERATED FILES:")
    submission_dir = Path("outputs/submission")
    if submission_dir.exists():
        for file in submission_dir.iterdir():
            if file.is_file():
                size = file.stat().st_size / 1024  # KB
                print(f"   📄 {file.name} ({size:.1f} KB)")
    
    print("\n💡 QUICK COMMANDS:")
    print("   🖼️  Open visualization: start outputs/submission/EuclideanTechnologies_AnomalyMap.png")
    print("   📊 Open metrics: start outputs/submission/EuclideanTechnologies_Metrics.xlsx")
    print("   🏃 Run training: python main.py --mode train --device cuda")
    print("   🔄 Batch inference: python main.py --mode batch_inference --input_dir data")
    print("   🎬 Video demo: python demo_system.py")
    
    print("\n" + "=" * 70)

def main():
    show_menu()
    
    print("\nWhat would you like to do next?")
    print("Type the number (1-7) or 'q' to quit:")
    
    choice = input("> ").strip()
    
    if choice == '1':
        print("\n🔍 Opening visualization files...")
        os.system("start outputs/submission/EuclideanTechnologies_AnomalyMap.png")
        os.system("start outputs/submission/EuclideanTechnologies_Metrics.xlsx")
    
    elif choice == '2':
        print("\n🚀 Starting full training...")
        print("Command: python main.py --mode train --device cuda")
        print("Note: This will take longer but produce better results")
    
    elif choice == '3':
        print("\n📊 Setting up batch processing...")
        print("Command: python main.py --mode batch_inference --input_dir data --output_dir outputs/batch")
    
    elif choice == '4':
        print("\n🎥 Video processing setup...")
        print("Command: python main.py --mode video_inference --input_path video.mp4")
    
    elif choice == '5':
        print("\n📈 Advanced analysis tools...")
        print("Command: python scripts/analyze_results.py")
    
    elif choice == '6':
        print("\n🔧 Customization options...")
        print("Edit config.yaml to modify parameters")
        print("Adjust thresholds, image sizes, model settings")
    
    elif choice == '7':
        print("\n📤 Export and packaging...")
        print("All files ready in outputs/submission/")
        print("Ready for submission or sharing")
    
    elif choice.lower() == 'q':
        print("\n👋 Goodbye!")
        return
    
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()