#!/usr/bin/env python3
"""
Evaluate Submission Script (ONNX Version)
==========================================

Evaluates both baseline_model.onnx and improved_model.onnx on the validation dataset.
Compares F1 scores and passes only if improved model performs better than baseline.

Usage:
    python evaluate_submission.py --model models/ --test-data ./test_data
"""

import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import logging
import sys
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import json
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Image preprocessing constants (must match training)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 150


def preprocess_image(image_path):
    """Load and preprocess a single image for ONNX inference."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to numpy and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = (img_array - MEAN) / STD
    
    # HWC to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def load_dataset(data_dir):
    """Load dataset from directory structure."""
    data_path = Path(data_dir)
    samples = []
    class_to_idx = {}
    
    # Get classes from directory structure (exclude hidden folders)
    classes = sorted([d.name for d in data_path.iterdir() 
                     if d.is_dir() and not d.name.startswith('.')])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    # Load all images
    for class_name in classes:
        class_dir = data_path / class_name
        class_idx = class_to_idx[class_name]
        
        for img_path in class_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                samples.append({
                    'path': str(img_path),
                    'label': class_idx,
                    'class_name': class_name
                })
    
    logger.info(f"Loaded {len(samples)} images from {len(classes)} classes")
    return samples, class_to_idx, idx_to_class


def load_onnx_model(model_path):
    """Load ONNX model for inference."""
    logger.info(f"Loading ONNX model from {model_path}")
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    return session


def evaluate_model(session, samples, class_names):
    """Evaluate ONNX model and return predictions and labels."""
    all_preds = []
    all_labels = []
    
    input_name = session.get_inputs()[0].name
    
    logger.info(f"Evaluating model on {len(samples)} samples...")
    for i, sample in enumerate(samples):
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(samples)} samples")
        
        # Preprocess image
        img_array = preprocess_image(sample['path'])
        
        # Run inference
        outputs = session.run(None, {input_name: img_array})
        logits = outputs[0]
        
        # Get prediction
        pred = np.argmax(logits, axis=1)[0]
        
        all_preds.append(pred)
        all_labels.append(sample['label'])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = 100 * accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'labels': all_labels
    }


def print_results(model_name, results, class_names):
    """Print evaluation results."""
    print("\n" + "="*70)
    print(f"{model_name.upper()} RESULTS")
    print("="*70)
    print(f"Accuracy:        {results['accuracy']:.2f}%")
    print(f"Macro F1 Score:  {results['f1_macro']:.4f}")
    print(f"Weighted F1:     {results['f1_weighted']:.4f}")
    
    print(f"\nPer-Class F1 Scores:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:15s}: {results['f1_per_class'][i]:.4f}")
    print("="*70 + "\n")


def compare_models(baseline_results, improved_results):
    """Compare baseline and improved models."""
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    baseline_f1 = baseline_results['f1_macro']
    improved_f1 = improved_results['f1_macro']
    
    baseline_acc = baseline_results['accuracy']
    improved_acc = improved_results['accuracy']
    
    f1_improvement = improved_f1 - baseline_f1
    acc_improvement = improved_acc - baseline_acc
    
    print(f"{'Metric':<20} {'Baseline':<15} {'Improved':<15} {'Change':<15}")
    print("-" * 70)
    print(f"{'Accuracy':<20} {baseline_acc:>6.2f}%{'':<8} {improved_acc:>6.2f}%{'':<8} {acc_improvement:>+6.2f}%{'':<8}")
    print(f"{'Macro F1 Score':<20} {baseline_f1:>6.4f}{'':<9} {improved_f1:>6.4f}{'':<9} {f1_improvement:>+6.4f}{'':<9}")
    print("="*70)
    
    if improved_f1 > baseline_f1:
        print(f"\n✅ PASS: Improved model F1 score ({improved_f1:.4f}) is better than baseline ({baseline_f1:.4f})")
        print(f"   Improvement: {f1_improvement:.4f} ({(f1_improvement/baseline_f1)*100:.2f}%)")
        return True
    else:
        print(f"\n❌ FAIL: Improved model F1 score ({improved_f1:.4f}) is NOT better than baseline ({baseline_f1:.4f})")
        print(f"   Difference: {f1_improvement:.4f} ({(f1_improvement/baseline_f1)*100:.2f}%)")
        return False


def save_results(baseline_results, improved_results, class_names, passed):
    """Save results to JSON file."""
    results = {
        'baseline': {
            'accuracy': float(baseline_results['accuracy']),
            'f1_macro': float(baseline_results['f1_macro']),
            'f1_weighted': float(baseline_results['f1_weighted']),
            'f1_per_class': {class_names[i]: float(baseline_results['f1_per_class'][i]) 
                            for i in range(len(class_names))}
        },
        'improved': {
            'accuracy': float(improved_results['accuracy']),
            'f1_macro': float(improved_results['f1_macro']),
            'f1_weighted': float(improved_results['f1_weighted']),
            'f1_per_class': {class_names[i]: float(improved_results['f1_per_class'][i]) 
                            for i in range(len(class_names))}
        },
        'comparison': {
            'f1_improvement': float(improved_results['f1_macro'] - baseline_results['f1_macro']),
            'accuracy_improvement': float(improved_results['accuracy'] - baseline_results['accuracy']),
            'passed': passed
        }
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to results.json")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate baseline and improved ONNX models on validation dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to models directory containing baseline_model.onnx and improved_model.onnx')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test/validation data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (not used in ONNX version, kept for compatibility)')
    
    args = parser.parse_args()
    
    logger.info("Using ONNX Runtime for evaluation")
    
    # Check if model files exist
    model_dir = Path(args.model)
    baseline_path = model_dir / 'baseline_model.onnx'
    improved_path = model_dir / 'improved_model.onnx'
    
    if not baseline_path.exists():
        logger.error(f"Baseline model not found at {baseline_path}")
        sys.exit(1)
    
    if not improved_path.exists():
        logger.error(f"Improved model not found at {improved_path}")
        sys.exit(1)
    
    # Load dataset
    logger.info(f"Loading test dataset from {args.test_data}")
    samples, class_to_idx, idx_to_class = load_dataset(args.test_data)
    
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    num_classes = len(class_names)
    
    logger.info(f"Found {num_classes} classes: {', '.join(class_names)}")
    
    # Evaluate baseline model
    logger.info("\n" + "="*70)
    logger.info("EVALUATING BASELINE MODEL")
    logger.info("="*70)
    baseline_session = load_onnx_model(baseline_path)
    baseline_results = evaluate_model(baseline_session, samples, class_names)
    print_results("Baseline Model", baseline_results, class_names)
    
    # Evaluate improved model
    logger.info("\n" + "="*70)
    logger.info("EVALUATING IMPROVED MODEL")
    logger.info("="*70)
    improved_session = load_onnx_model(improved_path)
    improved_results = evaluate_model(improved_session, samples, class_names)
    print_results("Improved Model", improved_results, class_names)
    
    # Compare models
    passed = compare_models(baseline_results, improved_results)
    
    # Save results
    save_results(baseline_results, improved_results, class_names, passed)
    
    # Exit with appropriate code
    if passed:
        logger.info("\n✅ Submission PASSED!")
        sys.exit(0)
    else:
        logger.error("\n❌ Submission FAILED!")
        sys.exit(1)


if __name__ == '__main__':
    main()
