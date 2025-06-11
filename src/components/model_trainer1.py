# -*- coding: utf-8 -*-
"""
Enhanced Water Body Segmentation with MLflow Multi-Model Tracking
"""

import os
import glob
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
from tqdm import tqdm

class Config:
    HEIGHT = 128
    WIDTH = 128
    BATCH_SIZE = 4
    EPOCHS = 5  # Reduced for quick testing
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_PATH = r"C:\Users\Administrator\Documents\water-bodies\data\processed"
    EXPERIMENT_NAME = "water_segmentation_multi_model"
    
    # Testing parameters
    NUM_WORKERS = 2
    PIN_MEMORY = False
    SUBSET_SIZE = 20  # Small subset for testing
    GRADIENT_ACCUMULATION = 4
    MIXED_PRECISION = False
    
    # MLflow settings
    MLFLOW_TRACKING_URI = "file:./mlruns"  # Local file storage
    # MLFLOW_TRACKING_URI = "http://localhost:5000"  # Use this for MLflow server

# Model configurations to experiment with
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

class WaterDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform or transforms.Compose([
            transforms.Resize((Config.HEIGHT, Config.WIDTH)),
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
            mask = transforms.Resize((Config.HEIGHT, Config.WIDTH))(mask)
            mask = torch.tensor(np.array(mask) / 255.0, dtype=torch.float32).unsqueeze(0)
            
            return image, mask
        except Exception as e:
            print(f"Error loading {self.images[idx]}: {e}")
            # Return a dummy sample if file is corrupted
            dummy_image = torch.zeros(3, Config.HEIGHT, Config.WIDTH)
            dummy_mask = torch.zeros(1, Config.HEIGHT, Config.WIDTH)
            return dummy_image, dummy_mask

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
    
    return model.to(Config.DEVICE)

def calculate_metrics(pred, target, threshold=0.5):
    """Calculate multiple metrics"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    # IoU
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    iou = (intersection / (union + 1e-8)).item()
    
    # Dice Score
    dice = (2 * intersection / (pred_binary.sum() + target.sum() + 1e-8)).item()
    
    # Pixel Accuracy
    correct = (pred_binary == target).sum().item()
    total = target.numel()
    pixel_acc = correct / total
    
    return {"iou": iou, "dice": dice, "pixel_accuracy": pixel_acc}

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_metrics = {"iou": 0, "dice": 0, "pixel_accuracy": 0}
    optimizer.zero_grad()
    
    for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Training")):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Gradient accumulation
        loss = loss / Config.GRADIENT_ACCUMULATION
        loss.backward()
        
        if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * Config.GRADIENT_ACCUMULATION
        
        # Calculate metrics
        batch_metrics = calculate_metrics(outputs, masks)
        for key in total_metrics:
            total_metrics[key] += batch_metrics[key]
        
        # Memory cleanup
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Average metrics
    avg_metrics = {key: value / len(dataloader) for key, value in total_metrics.items()}
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, avg_metrics

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_metrics = {"iou": 0, "dice": 0, "pixel_accuracy": 0}
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            # Calculate metrics
            batch_metrics = calculate_metrics(outputs, masks)
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]
    
    # Average metrics
    avg_metrics = {key: value / len(dataloader) for key, value in total_metrics.items()}
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, avg_metrics

def save_sample_predictions(model, val_loader, run_name, num_samples=3):
    """Save sample predictions as MLflow artifacts"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            image = images[0]
            mask = masks[0]
            pred = torch.sigmoid(model(image.unsqueeze(0).to(Config.DEVICE))).cpu()
            
            # Denormalize image for display
            img_display = image.permute(1, 2, 0).numpy()
            img_display = img_display * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            img_display = np.clip(img_display, 0, 1)
            
            axes[i, 0].imshow(img_display)
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask.squeeze(), cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred.squeeze(), cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    # Save plot as artifact
    plot_path = f"predictions_{run_name}.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    mlflow.log_artifact(plot_path)
    plt.close()
    
    # Clean up file
    if os.path.exists(plot_path):
        os.remove(plot_path)

def train_single_model(model_name, model_config, train_loader, val_loader):
    """Train a single model and track with MLflow"""
    
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # Log model configuration and hyperparameters
        mlflow.log_params({
            'model_name': model_name,
            'architecture': model_config['architecture'],
            'encoder_name': model_config['encoder_name'],
            'encoder_weights': model_config['encoder_weights'],
            'batch_size': Config.BATCH_SIZE,
            'epochs': Config.EPOCHS,
            'learning_rate': Config.LEARNING_RATE,
            'height': Config.HEIGHT,
            'width': Config.WIDTH,
            'subset_size': Config.SUBSET_SIZE,
            'device': str(Config.DEVICE)
        })
        
        # Create model
        model = create_model(model_config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_params({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        })
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        
        # Training loop
        best_iou = 0
        train_history = []
        val_history = []
        
        for epoch in range(Config.EPOCHS):
            print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
            
            # Train
            train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
            
            # Validate
            val_loss, val_metrics = validate_epoch(model, val_loader, criterion, Config.DEVICE)
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_iou': train_metrics['iou'],
                'train_dice': train_metrics['dice'],
                'train_pixel_accuracy': train_metrics['pixel_accuracy'],
                'val_loss': val_loss,
                'val_iou': val_metrics['iou'],
                'val_dice': val_metrics['dice'],
                'val_pixel_accuracy': val_metrics['pixel_accuracy']
            }, step=epoch)
            
            # Store history
            train_history.append({
                'epoch': epoch,
                'loss': train_loss,
                **train_metrics
            })
            val_history.append({
                'epoch': epoch,
                'loss': val_loss,
                **val_metrics
            })
            
            print(f"Train - Loss: {train_loss:.4f}, IoU: {train_metrics['iou']:.4f}, Dice: {train_metrics['dice']:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}")
            
            # Save best model
            if val_metrics['iou'] > best_iou:
                best_iou = val_metrics['iou']
                
                # Save model as artifact
                model_path = f"best_{model_name}_model.pth"
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(model_path)
                
                # Also log the model using MLflow's pytorch integration
                mlflow.pytorch.log_model(model, "pytorch_model")
                
                # Clean up local file
                if os.path.exists(model_path):
                    os.remove(model_path)
        
        # Log final metrics
        mlflow.log_metrics({
            'final_best_iou': best_iou,
            'final_train_loss': train_history[-1]['loss'],
            'final_val_loss': val_history[-1]['loss']
        })
        
        # Save training history as JSON artifact
        history = {
            'train_history': train_history,
            'val_history': val_history,
            'best_iou': best_iou
        }
        
        history_path = f"{model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        mlflow.log_artifact(history_path)
        
        # Save sample predictions
        save_sample_predictions(model, val_loader, model_name)
        
        # Clean up
        if os.path.exists(history_path):
            os.remove(history_path)
        
        print(f"Best IoU for {model_name}: {best_iou:.4f}")
        
        return {
            'model_name': model_name,
            'best_iou': best_iou,
            'final_train_loss': train_history[-1]['loss'],
            'final_val_loss': val_history[-1]['loss'],
            'run_id': mlflow.active_run().info.run_id
        }

def load_data():
    """Load and prepare data"""
    print("Loading data...")
    images = sorted(glob.glob(os.path.join(Config.DATA_PATH, 'images', '*')))
    masks = sorted(glob.glob(os.path.join(Config.DATA_PATH, 'masks', '*')))
    
    print(f"Found {len(images)} images and {len(masks)} masks")
    
    # Limit dataset to subset size if specified
    if Config.SUBSET_SIZE and Config.SUBSET_SIZE < len(images):
        images = images[:Config.SUBSET_SIZE]
        masks = masks[:Config.SUBSET_SIZE]
        print(f"Using subset of {len(images)} samples")
    
    # Split data
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}")
    
    # Create datasets
    train_dataset = WaterDataset(train_imgs, train_masks)
    val_dataset = WaterDataset(val_imgs, val_masks)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                             num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                           num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    
    return train_loader, val_loader

def run_experiments():
    """Run experiments with multiple models"""
    
    # Setup MLflow
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.EXPERIMENT_NAME)
    
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {Config.EXPERIMENT_NAME}")
    
    # Load data once for all experiments
    train_loader, val_loader = load_data()
    
    # Store results for comparison
    results = []
    
    # Train each model
    for model_name, model_config in MODEL_CONFIGS.items():
        try:
            result = train_single_model(model_name, model_config, train_loader, val_loader)
            results.append(result)
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    
    # Print comparison
    print(f"\n{'='*60}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*60}")
    
    # Sort by best IoU
    results.sort(key=lambda x: x['best_iou'], reverse=True)
    
    print(f"{'Model':<20} {'Best IoU':<10} {'Train Loss':<12} {'Val Loss':<10} {'Run ID'}")
    print("-" * 75)
    
    for result in results:
        print(f"{result['model_name']:<20} {result['best_iou']:<10.4f} "
              f"{result['final_train_loss']:<12.4f} {result['final_val_loss']:<10.4f} "
              f"{result['run_id'][:8]}...")
    
    print(f"\nBest model: {results[0]['model_name']} with IoU: {results[0]['best_iou']:.4f}")
    
    return results

def view_mlflow_ui_instructions():
    """Print instructions for viewing MLflow UI"""
    print(f"\n{'='*60}")
    print("HOW TO VIEW RESULTS IN MLFLOW UI")
    print(f"{'='*60}")
    print("1. Open terminal/command prompt")
    print("2. Navigate to your project directory")
    print("3. Run: mlflow ui")
    print("4. Open browser and go to: http://localhost:5000")
    print("5. You'll see all your experiments and can compare models!")
    print(f"{'='*60}")

def main():
    """Main function"""
    print("Starting Water Body Segmentation Multi-Model Experiment")
    print(f"Device: {Config.DEVICE}")
    print(f"Subset size: {Config.SUBSET_SIZE}")
    print(f"Models to test: {list(MODEL_CONFIGS.keys())}")
    
    # Run all experiments
    results = run_experiments()
    
    # Show instructions for viewing results
    view_mlflow_ui_instructions()
    
    return results

if __name__ == "__main__":
    main()