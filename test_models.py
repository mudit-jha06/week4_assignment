#!/usr/bin/env python3
"""
Test Script for evaluate_submission.py
======================================

This script allows you to test the evaluate_submission.py script locally
by providing custom paths to your dataset and model files.

Usage:
    python test_models.py \
        --baseline ./path/to/baseline_model.onnx \
        --improved ./path/to/improved_model.onnx \
        --test-data ./path/to/test_data
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil
import tempfile
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_file_exists(file_path, description):
    """Check if a file exists."""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"{description} not found at: {file_path}")
        return False
    logger.info(f"✓ Found {description}: {file_path}")
    return True


def check_directory_exists(dir_path, description):
    """Check if a directory exists."""
    path = Path(dir_path)
    if not path.exists() or not path.is_dir():
        logger.error(f"{description} not found at: {dir_path}")
        return False
    
    # Check if directory has subdirectories (classes)
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if not subdirs:
        logger.error(f"{description} has no class subdirectories")
        return False
    
    logger.info(f"✓ Found {description}: {dir_path}")
    logger.info(f"  Classes found: {', '.join([d.name for d in subdirs])}")
    return True


def setup_temp_models_dir(baseline_path, improved_path):
    """Create temporary directory with models in expected structure."""
    temp_dir = tempfile.mkdtemp(prefix='eval_test_')
    models_dir = Path(temp_dir) / 'models'
    models_dir.mkdir()
    
    baseline_path = Path(baseline_path)
    improved_path = Path(improved_path)
    
    # Copy ONNX models to temp directory
    shutil.copy2(baseline_path, models_dir / 'baseline_model.onnx')
    shutil.copy2(improved_path, models_dir / 'improved_model.onnx')
    
    # Also copy .onnx.data files if they exist (required for large models)
    baseline_data = baseline_path.parent / (baseline_path.stem + '.onnx.data')
    improved_data = improved_path.parent / (improved_path.stem + '.onnx.data')
    
    if baseline_data.exists():
        shutil.copy2(baseline_data, models_dir / 'baseline_model.onnx.data')
        logger.info(f"  Copied {baseline_data.name}")
    
    if improved_data.exists():
        shutil.copy2(improved_data, models_dir / 'improved_model.onnx.data')
        logger.info(f"  Copied {improved_data.name}")
    
    logger.info(f"Created temporary models directory: {models_dir}")
    return temp_dir, models_dir


def run_evaluation(models_dir, test_data_path, batch_size=32):
    """Run the evaluate_submission.py script."""
    script_path = Path(__file__).parent / 'week4_assignment' / 'evaluate_submission.py'
    
    if not script_path.exists():
        logger.error(f"evaluate_submission.py not found at: {script_path}")
        return None
    
    cmd = [
        'python',
        str(script_path),
        '--model', str(models_dir),
        '--test-data', str(test_data_path),
        '--batch-size', str(batch_size)
    ]
    
    logger.info("\n" + "="*70)
    logger.info("Running evaluation...")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("="*70 + "\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=False
        )
        return result.returncode
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Test evaluate_submission.py with custom model and data paths',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with specific models and dataset
  python test_models.py \\
      --baseline ./models/baseline_model.onnx \\
      --improved ./models/improved_model.onnx \\
      --test-data ./dataset/validation_data
  
  # Test with custom batch size
  python test_models.py \\
      --baseline ./models/baseline_model.onnx \\
      --improved ./models/improved_model.onnx \\
      --test-data ./validation_data \\
      --batch-size 64
        """
    )
    
    parser.add_argument('--baseline', type=str, required=True,
                       help='Path to baseline model (.onnx file)')
    parser.add_argument('--improved', type=str, required=True,
                       help='Path to improved model (.onnx file)')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test/validation data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("EVALUATE SUBMISSION TEST SCRIPT")
    print("="*70 + "\n")
    
    # Validate inputs
    logger.info("Validating inputs...")
    
    if not check_file_exists(args.baseline, "Baseline model"):
        sys.exit(1)
    
    if not check_file_exists(args.improved, "Improved model"):
        sys.exit(1)
    
    if not check_directory_exists(args.test_data, "Test data directory"):
        sys.exit(1)
    
    print()
    
    # Setup temporary directory structure
    temp_dir = None
    try:
        temp_dir, models_dir = setup_temp_models_dir(args.baseline, args.improved)
        
        # Run evaluation
        return_code = run_evaluation(models_dir, args.test_data, args.batch_size)
        
        if return_code is None:
            logger.error("Evaluation failed to run")
            sys.exit(1)
        
        # Print final result
        print("\n" + "="*70)
        if return_code == 0:
            print("✅ TEST PASSED - Improved model is better than baseline!")
            print("="*70 + "\n")
            sys.exit(0)
        else:
            print("❌ TEST FAILED - Improved model is NOT better than baseline")
            print("="*70 + "\n")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup temporary directory
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == '__main__':
    main()
