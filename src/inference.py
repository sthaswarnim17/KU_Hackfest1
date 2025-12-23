"""
Inference module for PM2.5 prediction with explainability.

Provides wrapper functions to load model and make predictions with visual explanations.
"""

import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, Tuple, Optional
import json

from model import load_model, ResNetPhysicsFusion
from features import extract_all_features, get_dark_channel_heatmap, dark_channel
from utils import (pm25_to_aqi_category, pm25_to_aqi_index, 
                   inverse_log_transform_pm25, load_normalization_stats)


class PM25Predictor:
    """
    Inference class for PM2.5 estimation from images.
    
    Provides prediction with explainability features.
    """
    
    def __init__(self, 
                 model_path: str,
                 normalization_path: str,
                 device: str = 'cpu'):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint (.pt)
            normalization_path: Path to feature normalization stats (JSON)
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        
        # Load normalization stats
        self.feature_mean, self.feature_std, self.feature_names = load_normalization_stats(
            normalization_path
        )
        
        # Load model - auto-detect architecture from checkpoint
        num_features = len(self.feature_names)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Detect model type from state_dict keys
        state_keys = list(checkpoint['model_state_dict'].keys())
        if any('efficientnet_backbone' in k for k in state_keys):
            from model import EfficientNetPhysicsFusion
            self.model = EfficientNetPhysicsFusion(num_features, dropout=0.3)
            print(f"Detected EfficientNet-B0 model")
        elif any('mobilenet_backbone' in k for k in state_keys):
            from model import MobileNetPhysicsFusion
            self.model = MobileNetPhysicsFusion(num_features, dropout=0.3)
            print(f"Detected MobileNetV2 model")
        else:
            from model import ResNet18PhysicsFusion
            self.model = ResNet18PhysicsFusion(num_features, dropout=0.3)
            print(f"Detected ResNet18 model")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Predictor initialized with {num_features} physics features")
    
    def preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for inference.
        
        Args:
            image: PIL Image (RGB)
            
        Returns:
            (image_tensor, image_array_224)
        """
        # Convert to numpy for physics features
        image_np = np.array(image)
        
        # Resize to 224x224 for feature extraction
        image_224 = image.resize((224, 224), Image.BILINEAR)
        image_np_224 = np.array(image_224)
        
        # Transform for model
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor, image_np_224
    
    def normalize_features(self, features_dict: Dict[str, float]) -> torch.Tensor:
        """
        Normalize physics features using training statistics.
        
        Args:
            features_dict: Dictionary of physics features
            
        Returns:
            Normalized feature tensor
        """
        # Extract features in correct order
        features_list = [features_dict[name] for name in self.feature_names]
        features = np.array(features_list, dtype=np.float32)
        
        # Normalize
        features = (features - self.feature_mean) / self.feature_std
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def predict(self, image: Image.Image) -> Dict:
        """
        Predict PM2.5 from image with explainability.
        
        Args:
            image: PIL Image (RGB)
            
        Returns:
            Dictionary with prediction results and explainability data
        """
        # Preprocess
        image_tensor, image_np = self.preprocess_image(image)
        
        # Extract physics features
        physics_features_dict = extract_all_features(image_np)
        physics_features = self.normalize_features(physics_features_dict)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        physics_features = physics_features.to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            log_pm25 = self.model(image_tensor, physics_features)
            log_pm25 = log_pm25.cpu().numpy()[0, 0]
        
        # Inverse transform
        pm25 = inverse_log_transform_pm25(log_pm25)
        pm25 = max(0, pm25)  # Ensure non-negative
        
        # Get AQI info
        aqi_category, aqi_color, health_advice = pm25_to_aqi_category(pm25)
        aqi_index = pm25_to_aqi_index(pm25)
        
        # Generate dark channel heatmap
        dark_channel_map = get_dark_channel_heatmap(image_np)
        
        # Get top contributing features
        top_features = self._get_top_features(physics_features_dict, n=5)
        
        # Compile results
        results = {
            'pm25': float(pm25),
            'aqi_index': int(aqi_index),
            'aqi_category': aqi_category,
            'aqi_color': aqi_color,
            'health_advice': health_advice,
            'physics_features': physics_features_dict,
            'top_features': top_features,
            'dark_channel_heatmap': dark_channel_map,
            'confidence': 'medium'  # Placeholder; can implement MC dropout for uncertainty
        }
        
        return results
    
    def _get_top_features(self, features_dict: Dict[str, float], n: int = 5) -> list:
        """
        Get top N physics features by absolute normalized value.
        
        Args:
            features_dict: Raw physics features
            n: Number of top features to return
            
        Returns:
            List of (feature_name, raw_value, normalized_value) tuples
        """
        # Normalize features
        normalized = {}
        for name in self.feature_names:
            idx = self.feature_names.index(name)
            raw_val = features_dict[name]
            norm_val = (raw_val - self.feature_mean[idx]) / self.feature_std[idx]
            normalized[name] = (raw_val, norm_val)
        
        # Sort by absolute normalized value
        sorted_features = sorted(normalized.items(), 
                                key=lambda x: abs(x[1][1]), 
                                reverse=True)
        
        # Format top N
        top_features = [
            {
                'name': name,
                'raw_value': float(raw_val),
                'normalized_value': float(norm_val)
            }
            for name, (raw_val, norm_val) in sorted_features[:n]
        ]
        
        return top_features
    
    def predict_from_path(self, image_path: str) -> Dict:
        """
        Predict PM2.5 from image file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Prediction results dictionary
        """
        image = Image.open(image_path).convert('RGB')
        return self.predict(image)
    
    def batch_predict(self, images: list) -> list:
        """
        Predict PM2.5 for multiple images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of prediction result dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results


def create_predictor(checkpoint_dir: str = './weights',
                    output_dir: str = './outputs',
                    device: str = 'cpu') -> PM25Predictor:
    """
    Create predictor with default paths.
    
    Args:
        checkpoint_dir: Directory containing model checkpoint
        output_dir: Directory containing normalization stats
        device: Device for inference
        
    Returns:
        PM25Predictor instance
    """
    model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    norm_path = os.path.join(output_dir, 'feature_normalization.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Normalization stats not found at {norm_path}")
    
    return PM25Predictor(model_path, norm_path, device=device)


def physics_only_baseline(image: Image.Image) -> Dict:
    """
    Physics-only baseline estimator (no ML model).
    
    Uses simple heuristics based on dark channel and contrast to estimate PM2.5.
    This serves as a fallback when model confidence is low.
    
    Args:
        image: PIL Image
        
    Returns:
        Baseline prediction dictionary
    """
    # Convert to numpy
    image_np = np.array(image.resize((224, 224)))
    
    # Extract physics features
    features = extract_all_features(image_np)
    
    # Simple heuristic: higher dark channel → higher PM2.5
    # This is a rough approximation
    dark_mean = features['dark_channel_mean']
    transmission_mean = features['transmission_mean']
    laplacian_var = features['laplacian_variance']
    contrast = features['contrast']
    
    # Rough formula (calibrated on typical ranges)
    # Higher dark channel, lower transmission → more haze → higher PM2.5
    pm25_estimate = (dark_mean / 255.0) * 150 + (1.0 - transmission_mean) * 50
    
    # Adjust for low sharpness (more haze)
    if laplacian_var < 50:
        pm25_estimate *= 1.2
    
    # Clamp to reasonable range
    pm25_estimate = np.clip(pm25_estimate, 0, 500)
    
    aqi_category, aqi_color, health_advice = pm25_to_aqi_category(pm25_estimate)
    aqi_index = pm25_to_aqi_index(pm25_estimate)
    
    return {
        'pm25': float(pm25_estimate),
        'aqi_index': int(aqi_index),
        'aqi_category': aqi_category,
        'aqi_color': aqi_color,
        'health_advice': health_advice,
        'method': 'physics_baseline',
        'note': 'This is a physics-only estimate without ML model'
    }


if __name__ == "__main__":
    # Test inference
    print("Testing PM2.5 predictor...")
    
    # Create dummy image
    dummy_image = Image.new('RGB', (640, 480), color=(150, 150, 150))
    
    # Test physics baseline
    print("\nTesting physics baseline...")
    baseline_result = physics_only_baseline(dummy_image)
    print(f"Baseline PM2.5: {baseline_result['pm25']:.2f} µg/m³")
    print(f"AQI Category: {baseline_result['aqi_category']}")
    
    # Test model predictor (will fail if model not trained yet)
    try:
        print("\nTesting model predictor...")
        predictor = create_predictor()
        result = predictor.predict(dummy_image)
        print(f"Model PM2.5: {result['pm25']:.2f} µg/m³")
        print(f"AQI Category: {result['aqi_category']}")
        print(f"Top features: {[f['name'] for f in result['top_features']]}")
    except FileNotFoundError as e:
        print(f"Model not found (expected before training): {e}")
    
    print("\nInference module test completed!")
