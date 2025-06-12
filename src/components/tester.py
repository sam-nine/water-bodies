# -*- coding: utf-8 -*-
"""
Water Body Segmentation Model Testing Script
"""

import os
import sys
import glob
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import json
from datetime import datetime
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
from tqdm import tqdm

# Add your project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.components.losses import BCEDiceLoss

class TestConfig:
    HEIGHT = 128
    WIDTH = 128
    BATCH_SIZE = 8  # Can be larger for testing since no gradients
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_PATH = r"C:\Users\Administrator\Documents\water-bodies\data\processed"
    
    # Testing parameters
    NUM_WORKERS = 2
    PIN_MEMORY = False
    TEST_SIZE = 0.2  # 20% of data for testing
    RANDOM_STATE = 42
    TEST_SUBSET_SIZE = 2000  # Limit test set to specific number of images
    
    # MLflow settings
    MLFLOW_TRACKING_URI = "file:./mlruns"
    EXPERIMENT_NAME = "water_segmentation_multi_model"
    
    # Output settings
    RESULTS_DIR = "test_results"
    SAVE_PREDICTIONS = True
    NUM_SAMPLE_PREDICTIONS = 10

# Model configurations (should match your training script)
MODEL_CONFIGS = {
    "unet_resnet34": {
        "architecture": "Unet",
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet"
    },
    "unet_resnet50": {
        "architecture": "Unet", 
        "encoder_name": "resnet50",
        "encoder_weights": "imagenet"
    },
    "deeplabv3_resnet34": {
        "architecture": "DeepLabV3",
        "encoder_name": "resnet34", 
        "encoder_weights": "imagenet"
    },
    "fpn_resnet34": {
        "architecture": "FPN",
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet"
    }
}

class WaterTestDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform or transforms.Compose([
            transforms.Resize((TestConfig.HEIGHT, TestConfig.WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            # Load and process image
            image = Image.open(self.images[idx]).convert('RGB')
            image = self.transform(image)
            
            # Load and process mask
            mask = Image.open(self.masks[idx]).convert('L')
            mask = transforms.Resize((TestConfig.HEIGHT, TestConfig.WIDTH))(mask)
            mask = torch.tensor(np.array(mask) / 255.0, dtype=torch.float32).unsqueeze(0)
            
            return image, mask, self.images[idx]  # Include filename for tracking
        except Exception as e:
            print(f"Error loading {self.images[idx]}: {e}")
            # Return a dummy sample if file is corrupted
            dummy_image = torch.zeros(3, TestConfig.HEIGHT, TestConfig.WIDTH)
            dummy_mask = torch.zeros(1, TestConfig.HEIGHT, TestConfig.WIDTH)
            return dummy_image, dummy_mask, self.images[idx]

def create_model(model_config):
    """Create a model based on configuration"""
    architecture = model_config["architecture"]
    encoder_name = model_config["encoder_name"] 
    encoder_weights = model_config["encoder_weights"]
    
    if architecture == "Unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1
        )
    elif architecture == "DeepLabV3":
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1
        )
    elif architecture == "FPN":
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model.to(TestConfig.DEVICE)

def calculate_detailed_metrics(pred, target, threshold=0.7):
    """Calculate comprehensive metrics for evaluation"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    # Convert to numpy for easier calculation
    pred_np = pred_binary.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # Flatten arrays
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    
    # Calculate basic metrics
    tp = np.sum((pred_flat == 1) & (target_flat == 1))
    tn = np.sum((pred_flat == 0) & (target_flat == 0))
    fp = np.sum((pred_flat == 1) & (target_flat == 0))
    fn = np.sum((pred_flat == 0) & (target_flat == 1))
    
    # IoU (Intersection over Union)
    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + 1e-8)
    
    # Dice Score (F1-Score for segmentation)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    
    # Pixel Accuracy
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    # Precision and Recall
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    # F1-Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Specificity
    specificity = tn / (tn + fp + 1e-8)
    
    return {
        "iou": iou,
        "dice": dice,
        "pixel_accuracy": pixel_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn
    }

def load_trained_model(model_name, model_config, model_path=None, run_id=None):
    """Load a trained model from file or MLflow"""
    model = create_model(model_config)
    
    if model_path and os.path.exists(model_path):
        # Load from local file
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=TestConfig.DEVICE))
    elif run_id:
        # Load from MLflow
        print(f"Loading model from MLflow run: {run_id}")
        mlflow.set_tracking_uri(TestConfig.MLFLOW_TRACKING_URI)
        model_uri = f"runs:/{run_id}/pytorch_model"
        model = mlflow.pytorch.load_model(model_uri, map_location=TestConfig.DEVICE)
    else:
        raise ValueError("Either model_path or run_id must be provided")
    
    model.eval()
    return model

def test_model(model, test_loader, model_name, criterion=None):
    """Test a single model and return detailed results"""
    if criterion is None:
        criterion = BCEDiceLoss()
    
    model.eval()
    all_metrics = []
    total_loss = 0
    predictions = []
    
    print(f"Testing {model_name}...")
    
    with torch.no_grad():
        for batch_idx, (images, masks, filenames) in enumerate(tqdm(test_loader, desc=f"Testing {model_name}")):
            images, masks = images.to(TestConfig.DEVICE), masks.to(TestConfig.DEVICE)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Calculate metrics for each sample in batch
            for i in range(images.size(0)):
                metrics = calculate_detailed_metrics(outputs[i:i+1], masks[i:i+1])
                metrics['filename'] = os.path.basename(filenames[i])
                all_metrics.append(metrics)
                
                # Store predictions for visualization
                if TestConfig.SAVE_PREDICTIONS and len(predictions) < TestConfig.NUM_SAMPLE_PREDICTIONS:
                    pred_sigmoid = torch.sigmoid(outputs[i]).cpu().numpy()
                    predictions.append({
                        'filename': filenames[i],
                        'image': images[i].cpu(),
                        'ground_truth': masks[i].cpu(),
                        'prediction': pred_sigmoid,
                        'metrics': metrics
                    })
    
    # Calculate average metrics
    avg_metrics = {}
    metric_keys = ['iou', 'dice', 'pixel_accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        avg_metrics[f'avg_{key}'] = np.mean(values)
        avg_metrics[f'std_{key}'] = np.std(values)
        avg_metrics[f'min_{key}'] = np.min(values)
        avg_metrics[f'max_{key}'] = np.max(values)
    
    avg_metrics['avg_loss'] = total_loss / len(test_loader)
    avg_metrics['num_samples'] = len(all_metrics)
    
    return avg_metrics, all_metrics, predictions

def save_test_predictions(predictions, model_name, results_dir):
    """Save sample predictions as images"""
    if not predictions:
        return
    
    pred_dir = os.path.join(results_dir, f"{model_name}_predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    # Create a grid of predictions
    num_samples = len(predictions)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, pred_data in enumerate(predictions):
        # Denormalize image for display
        image = pred_data['image']
        img_display = image.permute(1, 2, 0).numpy()
        img_display = img_display * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img_display = np.clip(img_display, 0, 1)
        
        # Display images
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title(f'Original\n{os.path.basename(pred_data["filename"])}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(pred_data['ground_truth'].squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_data['prediction'].squeeze(), cmap='gray')
        axes[i, 2].set_title(f'Prediction\nIoU: {pred_data["metrics"]["iou"]:.3f}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(pred_dir, f"{model_name}_test_predictions.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved predictions to {plot_path}")

def create_test_report(results, results_dir):
    """Create a comprehensive test report"""
    report_path = os.path.join(results_dir, "test_report.json")
    
    # Add timestamp
    results['test_timestamp'] = datetime.now().isoformat()
    results['test_config'] = {
        'device': str(TestConfig.DEVICE),
        'batch_size': TestConfig.BATCH_SIZE,
        'height': TestConfig.HEIGHT,
        'width': TestConfig.WIDTH,
        'test_size': TestConfig.TEST_SIZE
    }
    
    # Save as JSON
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create a summary text report
    summary_path = os.path.join(results_dir, "test_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("WATER BODY SEGMENTATION - TEST RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Date: {results['test_timestamp']}\n")
        f.write(f"Device: {TestConfig.DEVICE}\n")
        f.write(f"Number of test samples: {results.get('num_test_samples', 'N/A')}\n\n")
        
        # Model comparison
        if 'model_results' in results:
            f.write("MODEL PERFORMANCE COMPARISON:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Model':<20} {'IoU':<8} {'Dice':<8} {'F1':<8} {'Precision':<10} {'Recall':<8}\n")
            f.write("-" * 70 + "\n")
            
            for model_name, model_results in results['model_results'].items():
                f.write(f"{model_name:<20} "
                       f"{model_results['avg_iou']:<8.4f} "
                       f"{model_results['avg_dice']:<8.4f} "
                       f"{model_results['avg_f1_score']:<8.4f} "
                       f"{model_results['avg_precision']:<10.4f} "
                       f"{model_results['avg_recall']:<8.4f}\n")
        
        f.write("\nDetailed results saved in test_report.json\n")
    
    print(f"Test report saved to {report_path}")
    print(f"Summary saved to {summary_path}")

def load_test_data(subset_size=None, test_subset_size=None):
    """Load test data (unseen data not used in training)"""
    print("Loading test data...")
    
    # Load all data
    images = sorted(glob.glob(os.path.join(TestConfig.DATA_PATH, 'images', '*')))
    masks = sorted(glob.glob(os.path.join(TestConfig.DATA_PATH, 'masks', '*')))
    
    print(f"Found {len(images)} total images and {len(masks)} masks")
    
    # If subset_size was used in training, we need to split accordingly
    if subset_size and subset_size < len(images):
        # Use the same split as training to ensure no overlap
        train_imgs, remaining_imgs, train_masks, remaining_masks = train_test_split(
            images[:subset_size], masks[:subset_size], 
            test_size=TestConfig.TEST_SIZE, 
            random_state=TestConfig.RANDOM_STATE
        )
        
        # Use remaining images for testing (unseen data)
        test_images = remaining_imgs
        test_masks = remaining_masks
        
        # Also add some images that were never included in the subset
        if len(images) > subset_size:
            extra_images = images[subset_size:subset_size + len(remaining_imgs)]
            extra_masks = masks[subset_size:subset_size + len(remaining_masks)]
            test_images.extend(extra_images)
            test_masks.extend(extra_masks)
    else:
        # Standard split
        train_imgs, test_images, train_masks, test_masks = train_test_split(
            images, masks, 
            test_size=TestConfig.TEST_SIZE, 
            random_state=TestConfig.RANDOM_STATE
        )
    
    print(f"Available test images: {len(test_images)}")
    
    # Limit test set size if specified
    if test_subset_size and test_subset_size < len(test_images):
        # Randomly sample from test images
        test_indices = np.random.RandomState(TestConfig.RANDOM_STATE).choice(
            len(test_images), size=test_subset_size, replace=False
        )
        test_images = [test_images[i] for i in test_indices]
        test_masks = [test_masks[i] for i in test_indices]
        print(f"Using test subset of {len(test_images)} images")
    
    print(f"Final test set: {len(test_images)} images")
    
    # Create test dataset and loader
    test_dataset = WaterTestDataset(test_images, test_masks)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=TestConfig.BATCH_SIZE, 
        shuffle=False,
        num_workers=TestConfig.NUM_WORKERS, 
        pin_memory=TestConfig.PIN_MEMORY
    )
    
    return test_loader, len(test_images)

def test_all_models(model_paths=None, run_ids=None):
    """Test all models and create comparison"""
    
    # Create results directory
    os.makedirs(TestConfig.RESULTS_DIR, exist_ok=True)
    
    # Load test data
    test_loader, num_test_samples = load_test_data(subset_size=10000)  # Match your training subset
    
    results = {
        'model_results': {},
        'num_test_samples': num_test_samples
    }
    
    # Test each model
    for model_name, model_config in MODEL_CONFIGS.items():
        try:
            print(f"\n{'='*50}")
            print(f"Testing {model_name}")
            print(f"{'='*50}")
            
            # Determine how to load the model
            model_path = None
            run_id = None
            
            if model_paths and model_name in model_paths:
                model_path = model_paths[model_name]
            elif run_ids and model_name in run_ids:
                run_id = run_ids[model_name]
            else:
                print(f"No model path or run_id provided for {model_name}, skipping...")
                continue
            
            # Load and test model
            model = load_trained_model(model_name, model_config, model_path, run_id)
            avg_metrics, detailed_metrics, predictions = test_model(model, test_loader, model_name)
            
            # Store results
            results['model_results'][model_name] = avg_metrics
            
            # Save predictions
            if TestConfig.SAVE_PREDICTIONS:
                save_test_predictions(predictions, model_name, TestConfig.RESULTS_DIR)
            
            # Print results
            print(f"Results for {model_name}:")
            print(f"  Average IoU: {avg_metrics['avg_iou']:.4f} ± {avg_metrics['std_iou']:.4f}")
            print(f"  Average Dice: {avg_metrics['avg_dice']:.4f} ± {avg_metrics['std_dice']:.4f}")
            print(f"  Average F1: {avg_metrics['avg_f1_score']:.4f} ± {avg_metrics['std_f1_score']:.4f}")
            print(f"  Average Precision: {avg_metrics['avg_precision']:.4f} ± {avg_metrics['std_precision']:.4f}")
            print(f"  Average Recall: {avg_metrics['avg_recall']:.4f} ± {avg_metrics['std_recall']:.4f}")
            
            # Clean up GPU memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            continue
    
    # Create final report
    create_test_report(results, TestConfig.RESULTS_DIR)
    
    # Print final comparison
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS COMPARISON")
    print(f"{'='*60}")
    
    if results['model_results']:
        # Sort by IoU
        sorted_results = sorted(
            results['model_results'].items(), 
            key=lambda x: x[1]['avg_iou'], 
            reverse=True
        )
        
        print(f"{'Model':<20} {'IoU':<8} {'Dice':<8} {'F1':<8} {'Precision':<10} {'Recall':<8}")
        print("-" * 70)
        
        for model_name, model_results in sorted_results:
            print(f"{model_name:<20} "
                  f"{model_results['avg_iou']:<8.4f} "
                  f"{model_results['avg_dice']:<8.4f} "
                  f"{model_results['avg_f1_score']:<8.4f} "
                  f"{model_results['avg_precision']:<10.4f} "
                  f"{model_results['avg_recall']:<8.4f}")
        
        best_model = sorted_results[0]
        print(f"\nBest performing model: {best_model[0]} (IoU: {best_model[1]['avg_iou']:.4f})")
    
    return results

def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='Test Water Body Segmentation Models')
    parser.add_argument('--model_dir', type=str, help='Directory containing model files')
    parser.add_argument('--run_ids', type=str, nargs='+', help='MLflow run IDs to test (can be 1 or more)')
    parser.add_argument('--models', type=str, nargs='+', help='Specific model names to test with run_ids')
    parser.add_argument('--single_model', type=str, help='Test only a specific model')
    parser.add_argument('--model_path', type=str, help='Path to a specific model file')
    
    args = parser.parse_args()
    
    print("Starting Water Body Segmentation Model Testing")
    print(f"Device: {TestConfig.DEVICE}")
    print(f"Results will be saved to: {TestConfig.RESULTS_DIR}")
    
    # Setup model paths or run IDs
    model_paths = None
    run_ids = None
    
    if args.single_model and args.model_path:
        # Test single model
        if args.single_model not in MODEL_CONFIGS:
            print(f"Unknown model: {args.single_model}")
            print(f"Available models: {list(MODEL_CONFIGS.keys())}")
            return
        
        model_paths = {args.single_model: args.model_path}
    elif args.model_dir:
        # Look for model files in directory
        model_paths = {}
        for model_name in MODEL_CONFIGS.keys():
            model_file = os.path.join(args.model_dir, f"best_{model_name}_model.pth")
            if os.path.exists(model_file):
                model_paths[model_name] = model_file
    elif args.run_ids:
        # Use MLflow run IDs - can be any number
        print(f"Received {len(args.run_ids)} run IDs for testing")
        run_ids = {}
        
        if args.models:
            # User specified which models correspond to the run IDs
            if len(args.run_ids) != len(args.models):
                print(f"Number of run IDs ({len(args.run_ids)}) must match number of models ({len(args.models)})")
                return
            
            for model_name in args.models:
                if model_name not in MODEL_CONFIGS:
                    print(f"Unknown model: {model_name}")
                    print(f"Available models: {list(MODEL_CONFIGS.keys())}")
                    return
            
            run_ids = dict(zip(args.models, args.run_ids))
            print("Model-RunID mapping:")
            for model, rid in run_ids.items():
                print(f"  {model}: {rid}")
        else:
            # Auto-assign to first available models
            available_models = list(MODEL_CONFIGS.keys())
            for i, run_id in enumerate(args.run_ids):
                if i < len(available_models):
                    model_name = available_models[i]
                    run_ids[model_name] = run_id
                    print(f"Assigned run ID {run_id} to model {model_name}")
                else:
                    print(f"Warning: Extra run ID {run_id} ignored (no more models)")
                    break
    
    if not model_paths and not run_ids:
        print("No models specified for testing!")
        print("Examples:")
        print("  python testing.py --run_ids abc123 def456")
        print("  python testing.py --run_ids abc123 def456 --models unet_resnet34 unet_resnet50")
        print("  python testing.py --model_dir path/to/models")
        print("  python testing.py --single_model unet_resnet34 --model_path model.pth")
        return
    
    # Run tests
    results = test_all_models(model_paths, run_ids)
    
    print(f"\nTesting complete! Results saved to {TestConfig.RESULTS_DIR}")

if __name__ == "__main__":
    main()