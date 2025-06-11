# -*- coding: utf-8 -*-
"""
Simplified Water Body Segmentation with MLflow Tracking
"""

import os
import glob
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

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
    BATCH_SIZE = 4  # Reduced for laptop
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_PATH = r"C:\Users\Administrator\Documents\water-bodies\data\processed"
    EXPERIMENT_NAME = "water_segmentation_simple"
    
    # Laptop optimizations
    NUM_WORKERS = 2  # Reduced for laptop
    PIN_MEMORY = False  # Disable for CPU training
    SUBSET_SIZE = 10  # Use subset for faster training
    GRADIENT_ACCUMULATION = 4  # Simulate larger batch size
    MIXED_PRECISION = False  # Set True if you have GPU with tensor cores

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

def calculate_iou(pred, target, threshold=0.5):
    """Calculate IoU metric"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    return (intersection / (union + 1e-8)).item()

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_iou = 0
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
        total_iou += calculate_iou(outputs, masks)
        
        # Memory cleanup
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return total_loss / len(dataloader), total_iou / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            total_iou += calculate_iou(outputs, masks)
    
    return total_loss / len(dataloader), total_iou / len(dataloader)

def train_model():
    # Setup MLflow
    mlflow.set_experiment(Config.EXPERIMENT_NAME)
    
    with mlflow.start_run():
        # Log config
        mlflow.log_params({
            'batch_size': Config.BATCH_SIZE,
            'epochs': Config.EPOCHS,
            'learning_rate': Config.LEARNING_RATE,
            'height': Config.HEIGHT,
            'width': Config.WIDTH
        })
        
        # Load data
        images = sorted(glob.glob(os.path.join(Config.DATA_PATH, 'images', '*')))
        masks = sorted(glob.glob(os.path.join(Config.DATA_PATH, 'masks', '*')))
        
        # Limit dataset to subset size if specified
        if Config.SUBSET_SIZE and Config.SUBSET_SIZE < len(images):
            images = images[:Config.SUBSET_SIZE]
            masks = masks[:Config.SUBSET_SIZE]
            
        # Split data
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            images, masks, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = WaterDataset(train_imgs, train_masks)
        val_dataset = WaterDataset(val_imgs, val_masks)
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        # Create model - Using single UNet model as requested
        model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1
        ).to(Config.DEVICE)
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        
        # Training loop
        best_iou = 0
        for epoch in range(Config.EPOCHS):
            print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
            
            train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
            val_loss, val_iou = validate_epoch(model, val_loader, criterion, Config.DEVICE)
            
            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_iou': train_iou,
                'val_loss': val_loss,
                'val_iou': val_iou
            }, step=epoch)
            
            print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            
            # Save best model
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), 'best_water_model.pth')
                mlflow.pytorch.log_model(model, "model")
        
        print(f"\nBest IoU: {best_iou:.4f}")
        return model

def visualize_predictions(model, val_loader, num_samples=3):
    """Simple visualization function"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
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
    plt.show()

def main():
    """Main training function"""
    print("Starting simplified water body segmentation training...")
    model = train_model()
    
    # Optional: Load data for visualization
    images = sorted(glob.glob(os.path.join(Config.DATA_PATH, 'images', '*')))
    masks = sorted(glob.glob(os.path.join(Config.DATA_PATH, 'masks', '*')))
    _, val_imgs, _, val_masks = train_test_split(images, masks, test_size=0.2, random_state=42)
    
    val_dataset = WaterDataset(val_imgs, val_masks)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    visualize_predictions(model, val_loader)

if __name__ == "__main__":
    main()