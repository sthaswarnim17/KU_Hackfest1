"""
Model architecture: MobileNetV2 + Physics Features Fusion for PM2.5 estimation.

Combines pretrained MobileNetV2 visual features with handcrafted physics features.
MobileNetV2 is lighter and faster than ResNet while maintaining good accuracy.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class MobileNetPhysicsFusion(nn.Module):
    """
    Fusion model combining MobileNetV2 vision backbone with physics features.
    
    Architecture:
        - MobileNetV2 (pretrained on ImageNet) → 1280-dim embedding
        - Physics features → direct input
        - Concat → MLP → PM2.5 prediction
    
    MobileNetV2 advantages:
        - Faster inference (fewer parameters)
        - Lower memory usage
        - Good feature extraction for outdoor scenes
    """
    
    def __init__(self, 
                 num_physics_features: int,
                 freeze_backbone: bool = False,
                 dropout: float = 0.3):
        """
        Initialize fusion model.
        
        Args:
            num_physics_features: Number of physics feature inputs
            freeze_backbone: If True, freeze early MobileNet layers
            dropout: Dropout probability in fusion MLP
        """
        super(MobileNetPhysicsFusion, self).__init__()
        
        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        # Remove final classifier (keep features only)
        self.mobilenet_backbone = mobilenet.features
        
        # Add adaptive pooling to get fixed-size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Optionally freeze early layers
        if freeze_backbone:
            # Freeze first 10 layers (out of 18 feature layers)
            for i, module in enumerate(self.mobilenet_backbone):
                if i < 10:
                    for param in module.parameters():
                        param.requires_grad = False
        
        # MobileNetV2 produces 1280-dim features
        self.mobilenet_output_dim = 1280
        
        # Fusion MLP
        # Input: 1280 (MobileNet) + num_physics_features
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.mobilenet_output_dim + num_physics_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Single output: PM2.5
        )
        
    def forward(self, image: torch.Tensor, physics_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            image: Image tensor (B, 3, 224, 224)
            physics_features: Physics features tensor (B, num_physics_features)
            
        Returns:
            predictions: PM2.5 predictions (B, 1)
        """
        # Extract visual features
        visual_features = self.mobilenet_backbone(image)  # (B, 1280, 7, 7)
        visual_features = self.adaptive_pool(visual_features)  # (B, 1280, 1, 1)
        visual_features = visual_features.view(visual_features.size(0), -1)  # (B, 1280)
        
        # Concatenate visual and physics features
        combined = torch.cat([visual_features, physics_features], dim=1)
        
        # Predict PM2.5
        predictions = self.fusion_mlp(combined)
        
        return predictions
    
    def get_visual_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features for analysis/visualization.
        
        Args:
            image: Image tensor (B, 3, 224, 224)
            
        Returns:
            visual_features: Feature vector (B, 1280)
        """
        with torch.no_grad():
            visual_features = self.mobilenet_backbone(image)
            visual_features = self.adaptive_pool(visual_features)
            visual_features = visual_features.view(visual_features.size(0), -1)
        return visual_features


# Alias for backward compatibility
ResNetPhysicsFusion = MobileNetPhysicsFusion


class EfficientNetPhysicsFusion(nn.Module):
    """
    EfficientNet-B0 fusion model - State-of-the-art accuracy.
    
    Uses compound scaling for better feature extraction than MobileNetV2.
    """
    
    def __init__(self, 
                 num_physics_features: int,
                 freeze_backbone: bool = False,
                 dropout: float = 0.3):
        super(EfficientNetPhysicsFusion, self).__init__()
        
        # Load pretrained EfficientNet-B0
        efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.efficientnet_backbone = efficientnet.features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if freeze_backbone:
            for i, module in enumerate(self.efficientnet_backbone):
                if i < 5:
                    for param in module.parameters():
                        param.requires_grad = False
        
        self.efficientnet_output_dim = 1280
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.efficientnet_output_dim + num_physics_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
    def forward(self, image: torch.Tensor, physics_features: torch.Tensor) -> torch.Tensor:
        visual_features = self.efficientnet_backbone(image)
        visual_features = self.adaptive_pool(visual_features)
        visual_features = visual_features.view(visual_features.size(0), -1)
        combined = torch.cat([visual_features, physics_features], dim=1)
        return self.fusion_mlp(combined)
    
    def get_visual_features(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            visual_features = self.efficientnet_backbone(image)
            visual_features = self.adaptive_pool(visual_features)
            return visual_features.view(visual_features.size(0), -1)


class ResNet18PhysicsFusion(nn.Module):
    """
    Original ResNet18 fusion model (kept for comparison).
    """
    
    def __init__(self,
                 num_physics_features: int,
                 freeze_backbone: bool = True,
                 dropout: float = 0.3):
        """
        Initialize ResNet18 fusion model.
        
        Args:
            num_physics_features: Number of physics feature inputs
            freeze_backbone: If True, freeze early layers
            dropout: Dropout probability
        """
        super(ResNet18PhysicsFusion, self).__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Optionally freeze early layers
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'layer4' not in name:
                    param.requires_grad = False
        
        self.backbone_output_dim = 512
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.backbone_output_dim + num_physics_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, image: torch.Tensor, physics_features: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        visual_features = self.backbone(image)
        visual_features = visual_features.view(visual_features.size(0), -1)
        
        combined = torch.cat([visual_features, physics_features], dim=1)
        predictions = self.fusion_mlp(combined)
        
        return predictions
    
    def get_visual_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract visual features."""
        with torch.no_grad():
            visual_features = self.backbone(image)
            visual_features = visual_features.view(visual_features.size(0), -1)
        return visual_features


def count_parameters(model: nn.Module) -> tuple:
    """
    Count trainable and total parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    return trainable, total


def save_model(model: nn.Module, 
               filepath: str,
               metadata: Optional[dict] = None):
    """
    Save model checkpoint with metadata.
    
    Args:
        model: Model to save
        filepath: Path to save .pt file
        metadata: Optional dictionary with training info
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': model.__class__.__name__,
    }
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str,
               num_physics_features: int,
               model_type: str = 'MobileNetPhysicsFusion',
               device: str = 'cpu') -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        filepath: Path to .pt file
        num_physics_features: Number of physics features
        model_type: Model class name
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    # Initialize model
    if model_type in ['MobileNetPhysicsFusion', 'ResNetPhysicsFusion']:
        model = MobileNetPhysicsFusion(num_physics_features)
    elif model_type == 'ResNet18PhysicsFusion':
        model = ResNet18PhysicsFusion(num_physics_features)
    else:
        # Try MobileNet as default
        model = MobileNetPhysicsFusion(num_physics_features)
    
    # Load weights
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {filepath}")
    
    if 'metadata' in checkpoint:
        print("Checkpoint metadata:", checkpoint['metadata'])
    
    return model


if __name__ == "__main__":
    # Test model
    print("Testing ResNetPhysicsFusion model...")
    
    # Create dummy data
    batch_size = 4
    num_physics_features = 24
    
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_physics = torch.randn(batch_size, num_physics_features)
    
    # Initialize model
    model = ResNetPhysicsFusion(
        num_physics_features=num_physics_features,
        freeze_backbone=True,
        dropout=0.3
    )
    
    print(f"\nModel: {model.__class__.__name__}")
    
    # Count parameters
    trainable, total = count_parameters(model)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(dummy_images, dummy_physics)
    
    print(f"\nInput shapes:")
    print(f"  Images: {dummy_images.shape}")
    print(f"  Physics features: {dummy_physics.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:3, 0].tolist()}")
    
    # Test visual feature extraction
    visual_feats = model.get_visual_features(dummy_images)
    print(f"\nVisual features shape: {visual_feats.shape}")
    
    print("\nModel test completed successfully!")
