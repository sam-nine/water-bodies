# -*- coding: utf-8 -*-
"""
Water Body Segmentation Inference Service
src/components/inference_service.py
"""

import os
import sys
import glob
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import mlflow
import mlflow.pytorch
from typing import Union, List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.logger import logging
from src.exception import CustomException

class WaterSegmentationInference:
    """
    Inference service for water body segmentation.
    Loads trained models and processes new images for water detection.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 mlflow_model_uri: Optional[str] = None,
                 device: Optional[str] = None,
                 confidence_threshold: float = 0.5):
        """
        Initialize the inference service.
        
        Args:
            model_path: Path to saved PyTorch model (.pth file)
            mlflow_model_uri: MLflow model URI (e.g., "runs:/RUN_ID/pytorch_model")
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
            confidence_threshold: Threshold for binary segmentation
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_config = None
        self.transform = self._setup_transforms()
        
        logging.info(f"Initializing inference service on device: {self.device}")
        
        # Load model
        if mlflow_model_uri:
            self.load_model_from_mlflow(mlflow_model_uri)
        elif model_path:
            self.load_model_from_path(model_path)
        else:
            logging.warning("No model specified. Use load_model_* methods to load a model.")
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((128, 128)),  # Match training size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_model_from_mlflow(self, model_uri: str):
        """
        Load model from MLflow.
        
        Args:
            model_uri: MLflow model URI (e.g., "runs:/RUN_ID/pytorch_model")
        """
        try:
            logging.info(f"Loading model from MLflow: {model_uri}")
            
            # Load model
            self.model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
            self.model.eval()
            
            # Try to load model config from MLflow run
            if model_uri.startswith("runs:/"):
                run_id = model_uri.split("/")[1]
                run = mlflow.get_run(run_id)
                self.model_config = {
                    'architecture': run.data.params.get('architecture', 'Unknown'),
                    'encoder_name': run.data.params.get('encoder_name', 'Unknown'),
                    'model_name': run.data.params.get('model_name', 'Unknown')
                }
            
            logging.info("Model loaded successfully from MLflow")
            
        except Exception as e:
            logging.error(f"Error loading model from MLflow: {e}")
            raise CustomException(e, sys)
    
    #not using this
    def load_model_from_path(self, model_path: str, model_config: Optional[Dict] = None):
        """
        Load model from local path.
        
        Args:
            model_path: Path to saved model weights (.pth file)
            model_config: Model configuration dict with architecture details
        """
        try:
            logging.info(f"Loading model from path: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # If no config provided, try to infer from filename or use default
            if model_config is None:
                model_config = self._infer_model_config_from_path(model_path)
            
            # Create model architecture
            self.model = self._create_model_from_config(model_config)
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            self.model_config = model_config
            logging.info("Model loaded successfully from path")
            
        except Exception as e:
            logging.error(f"Error loading model from path: {e}")
            raise CustomException(e, sys)
    
    def _infer_model_config_from_path(self, model_path: str) -> Dict:
        """Infer model config from filename or return default"""
        filename = os.path.basename(model_path).lower()
        
        # Try to extract info from filename
        if 'unet' in filename and 'resnet34' in filename:
            return {
                'architecture': 'Unet',
                'encoder_name': 'resnet34',
                'encoder_weights': 'imagenet'
            }
        elif 'unet' in filename and 'resnet50' in filename:
            return {
                'architecture': 'Unet',
                'encoder_name': 'resnet50', 
                'encoder_weights': 'imagenet'
            }
        elif 'deeplabv3' in filename:
            return {
                'architecture': 'DeepLabV3',
                'encoder_name': 'resnet34',
                'encoder_weights': 'imagenet'
            }
        elif 'fpn' in filename:
            return {
                'architecture': 'FPN',
                'encoder_name': 'resnet34',
                'encoder_weights': 'imagenet'
            }
        else:
            # Default fallback
            logging.warning("Could not infer model config from filename, using default UNet")
            return {
                'architecture': 'Unet',
                'encoder_name': 'resnet34',
                'encoder_weights': 'imagenet'
            }
    
    def _create_model_from_config(self, config: Dict):
        """Create model from configuration"""
        architecture = config['architecture']
        encoder_name = config['encoder_name']
        encoder_weights = config['encoder_weights']
        
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
        
        return model.to(self.device)
    
    def preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess input image for inference.
        
        Args:
            image_input: Path to image file, numpy array, or PIL Image
            
        Returns:
            Preprocessed tensor ready for model input
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # Load from file path
                image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, np.ndarray):
                # Convert numpy array to PIL
                image = Image.fromarray(image_input.astype(np.uint8)).convert('RGB')
            elif isinstance(image_input, Image.Image):
                # Already PIL Image
                image = image_input.convert('RGB')
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Apply transforms
            tensor = self.transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logging.error(f"Error preprocessing image: {e}")
            raise CustomException(e, sys)
    
    def postprocess_output(self, model_output: torch.Tensor, 
                          original_size: Optional[Tuple[int, int]] = None) -> Dict:
        """
        Postprocess model output to generate final segmentation mask.
        
        Args:
            model_output: Raw model output tensor
            original_size: Original image size (width, height) for resizing mask
            
        Returns:
            Dictionary containing processed results
        """
        try:
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(model_output).cpu().numpy()
            
            # Remove batch dimension
            prob_mask = probs.squeeze()
            
            # Apply threshold to get binary mask
            binary_mask = (prob_mask > self.confidence_threshold).astype(np.uint8)
            
            # Resize to original size if provided
            if original_size:
                prob_mask = cv2.resize(prob_mask, original_size, interpolation=cv2.INTER_LINEAR)
                binary_mask = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
            
            # Calculate confidence statistics
            water_pixels = np.sum(binary_mask)
            total_pixels = binary_mask.size
            water_percentage = (water_pixels / total_pixels) * 100
            avg_confidence = np.mean(prob_mask[binary_mask == 1]) if water_pixels > 0 else 0.0
            
            return {
                'probability_mask': prob_mask,
                'binary_mask': binary_mask,
                'water_percentage': water_percentage,
                'water_pixels': int(water_pixels),
                'total_pixels': int(total_pixels),
                'average_confidence': float(avg_confidence),
                'max_confidence': float(np.max(prob_mask)),
                'min_confidence': float(np.min(prob_mask))
            }
            
        except Exception as e:
            logging.error(f"Error postprocessing output: {e}")
            raise CustomException(e, sys)
    
    def predict_single_image(self, image_input: Union[str, np.ndarray, Image.Image],
                           return_original_size: bool = True) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image_input: Input image (path, numpy array, or PIL Image)
            return_original_size: Whether to resize output to original image size
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("No model loaded. Use load_model_* methods first.")
        
        try:
            # Get original size if needed
            original_size = None
            if return_original_size:
                if isinstance(image_input, str):
                    with Image.open(image_input) as img:
                        original_size = img.size  # (width, height)
                elif isinstance(image_input, np.ndarray):
                    original_size = (image_input.shape[1], image_input.shape[0])
                elif isinstance(image_input, Image.Image):
                    original_size = image_input.size
            
            # Preprocess
            input_tensor = self.preprocess_image(image_input)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Postprocess
            results = self.postprocess_output(output, original_size)
            
            # Add metadata
            results['model_config'] = self.model_config
            results['confidence_threshold'] = self.confidence_threshold
            results['timestamp'] = datetime.now().isoformat()
            
            logging.info(f"Inference completed. Water coverage: {results['water_percentage']:.2f}%")
            
            return results
            
        except Exception as e:
            logging.error(f"Error during inference: {e}")
            raise CustomException(e, sys)
    
    def predict_batch(self, image_list: List[Union[str, np.ndarray, Image.Image]],
                     batch_size: int = 4) -> List[Dict]:
        """
        Run inference on multiple images in batches.
        
        Args:
            image_list: List of input images
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        if self.model is None:
            raise ValueError("No model loaded. Use load_model_* methods first.")
        
        results = []
        
        for i in range(0, len(image_list), batch_size):
            batch = image_list[i:i + batch_size]
            batch_results = []
            
            for image in batch:
                try:
                    result = self.predict_single_image(image)
                    batch_results.append(result)
                except Exception as e:
                    logging.error(f"Error processing image in batch: {e}")
                    batch_results.append({
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
            
            results.extend(batch_results)
            logging.info(f"Processed batch {i//batch_size + 1}/{(len(image_list)-1)//batch_size + 1}")
        
        return results
    
    def save_prediction_visualization(self, image_input: Union[str, np.ndarray, Image.Image],
                                    output_path: str, prediction_result: Optional[Dict] = None):
        """
        Save a visualization of the prediction results.
        
        Args:
            image_input: Original input image
            output_path: Path to save the visualization
            prediction_result: Pre-computed prediction results (optional)
        """
        try:
            # Get prediction if not provided
            if prediction_result is None:
                prediction_result = self.predict_single_image(image_input)
            
            # Load original image
            if isinstance(image_input, str):
                original_img = np.array(Image.open(image_input).convert('RGB'))
            elif isinstance(image_input, np.ndarray):
                original_img = image_input
            elif isinstance(image_input, Image.Image):
                original_img = np.array(image_input.convert('RGB'))
            
            # Get masks
            prob_mask = prediction_result['probability_mask']
            binary_mask = prediction_result['binary_mask']
            
            # Resize masks to match original image if needed
            if original_img.shape[:2] != prob_mask.shape:
                prob_mask = cv2.resize(prob_mask, (original_img.shape[1], original_img.shape[0]))
                binary_mask = cv2.resize(binary_mask, (original_img.shape[1], original_img.shape[0]))
            
            # Create visualization
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Original image
            axes[0, 0].imshow(original_img)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Probability mask
            axes[0, 1].imshow(prob_mask, cmap='Blues', vmin=0, vmax=1)
            axes[0, 1].set_title(f'Water Probability\n(Max: {prediction_result["max_confidence"]:.3f})')
            axes[0, 1].axis('off')
            
            # Binary mask
            axes[1, 0].imshow(binary_mask, cmap='Blues', vmin=0, vmax=1)
            axes[1, 0].set_title(f'Water Mask\n({prediction_result["water_percentage"]:.1f}% water)')
            axes[1, 0].axis('off')
            
            # Overlay
            overlay = original_img.copy()
            water_overlay = np.zeros_like(original_img)
            water_overlay[:, :, 2] = binary_mask * 255  # Blue channel
            overlay = cv2.addWeighted(overlay, 0.7, water_overlay, 0.3, 0)
            
            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title('Water Overlay')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Visualization saved to: {output_path}")
            
        except Exception as e:
            logging.error(f"Error saving visualization: {e}")
            raise CustomException(e, sys)



def main():
    # === ARGUMENT PARSING ===
    parser = argparse.ArgumentParser(description="Segment water bodies in satellite images.")
    parser.add_argument('--input', type=str, required=True, help="Path to input image or folder")
    parser.add_argument('--output', type=str, default="output", help="Path to save output")
    parser.add_argument('--model_uri', type=str, default="runs:/d792473199e2474d8ba76b89ef32c62e/pytorch_model", help="MLflow model URI")
    parser.add_argument('--threshold', type=float, default=0.7, help="Confidence threshold")

    args = parser.parse_args()
    input_path = args.input
    output_dir = args.output
    model_uri = args.model_uri
    confidence_threshold = args.threshold
    # ==========================

    os.makedirs(output_dir, exist_ok=True)


    inference_service = WaterSegmentationInference(
        mlflow_model_uri=model_uri,
        confidence_threshold=confidence_threshold
    )

    def process_and_save(image_path: str):
        try:
            image_name = Path(image_path).stem
            result = inference_service.predict_single_image(image_path)

           # Create a subfolder for this image
            image_output_dir = os.path.join(output_dir, image_name)
            os.makedirs(image_output_dir, exist_ok=True)

            # Save original image
            original = Image.open(image_path).convert("RGB")
            original.save(os.path.join(image_output_dir, "original.jpg"))

            # Save binary mask
            binary_mask = (result["binary_mask"] * 255).astype(np.uint8)
            Image.fromarray(binary_mask).save(os.path.join(image_output_dir, "mask.jpg"))


            logging.info(f"Processed: {image_path} -> Output saved in {output_dir}")
        except Exception as e:
            logging.error(f"Failed to process {image_path}: {e}")

    # Determine if input is file or folder
    if os.path.isdir(input_path):
        image_files = sorted(glob.glob(os.path.join(input_path, "*.jpg")) +
                             glob.glob(os.path.join(input_path, "*.png")))
        for image_file in image_files:
            process_and_save(image_file)
    elif os.path.isfile(input_path):
        process_and_save(input_path)
    else:
        logging.error("Input path is not valid.")


if __name__ == "__main__":
    main()
