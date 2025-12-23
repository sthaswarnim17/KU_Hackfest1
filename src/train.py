"""
Training script for Aero-Gauge PM2.5 estimation model.

Trains ResNet+Physics fusion model with logging, checkpointing, and evaluation.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from dataset import PM25VisionDataset, get_train_transforms, get_val_transforms, create_dataloaders
from model import EfficientNetPhysicsFusion as ResNetPhysicsFusion, count_parameters, save_model
from utils import (compute_metrics, log_transform_pm25, inverse_log_transform_pm25,
                   normalize_features, save_normalization_stats, print_metrics_summary,
                   stratified_split, get_physics_baseline)


class Trainer:
    """Training manager for PM2.5 estimation model."""
    
    def __init__(self, args):
        """Initialize trainer with arguments."""
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Setup data
        self.setup_data()
        
        # Setup model
        self.setup_model()
        
        # Setup training
        self.setup_training()
        
        # Tracking
        self.best_val_mae = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        self.patience_counter = 0
        
    def setup_data(self):
        """Setup datasets and dataloaders."""
        print("\n=== Setting up data ===")
        
        # Create split CSVs if not exists
        if not all([os.path.exists(f"{self.args.data_dir}/{split}_split.csv") 
                   for split in ['train', 'val', 'test']]):
            print("Creating train/val/test splits...")
            self.create_splits()
        
        # Create dataloaders
        train_csv = f"{self.args.data_dir}/train_split.csv"
        val_csv = f"{self.args.data_dir}/val_split.csv"
        test_csv = f"{self.args.data_dir}/test_split.csv"
        
        # All splits use train/images directory (since we split the training data)
        train_img_dir = f"{self.args.data_dir}/train/images"
        val_img_dir = f"{self.args.data_dir}/train/images"
        test_img_dir = f"{self.args.data_dir}/train/images"
        
        self.train_loader, self.val_loader, self.test_loader, self.feature_names = create_dataloaders(
            train_csv, val_csv, test_csv,
            train_img_dir, val_img_dir, test_img_dir,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            cache_dir=self.args.cache_dir
        )
        
        self.num_physics_features = len(self.feature_names)
        print(f"Number of physics features: {self.num_physics_features}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Test batches: {len(self.test_loader)}")
        
        # Normalize physics features
        self.normalize_physics_features()
        
    def create_splits(self):
        """Create train/val/test split CSV files."""
        # Load full metadata
        train_metadata = pd.read_csv(f"{self.args.data_dir}/train/metadata.csv")
        
        # Stratified split
        train_df, val_df, test_df = stratified_split(
            train_metadata,
            pm25_column='pm2_5' if 'pm2_5' in train_metadata.columns else 'pm25',
            train_ratio=0.8,
            val_ratio=0.1,
            random_state=42
        )
        
        # Save splits
        train_df.to_csv(f"{self.args.data_dir}/train_split.csv", index=False)
        val_df.to_csv(f"{self.args.data_dir}/val_split.csv", index=False)
        test_df.to_csv(f"{self.args.data_dir}/test_split.csv", index=False)
        
        print(f"Created splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    def normalize_physics_features(self):
        """Compute normalization stats for physics features."""
        # Collect all physics features from training set
        train_features = []
        for _, physics_features, _ in self.train_loader:
            train_features.append(physics_features.numpy())
        
        train_features = np.concatenate(train_features, axis=0)
        
        # Compute mean and std
        self.feature_mean = torch.tensor(np.mean(train_features, axis=0), dtype=torch.float32)
        self.feature_std = torch.tensor(np.std(train_features, axis=0), dtype=torch.float32)
        self.feature_std[self.feature_std < 1e-8] = 1.0
        
        # Save normalization stats
        save_normalization_stats(
            self.feature_mean.numpy(),
            self.feature_std.numpy(),
            self.feature_names,
            f"{self.args.output_dir}/feature_normalization.json"
        )
        
        print("Computed feature normalization statistics")
        
    def setup_model(self):
        """Initialize model."""
        print("\n=== Setting up model ===")
        
        self.model = ResNetPhysicsFusion(
            num_physics_features=self.num_physics_features,
            freeze_backbone=self.args.freeze_backbone,
            dropout=self.args.dropout
        )
        
        self.model.to(self.device)
        
        trainable, total = count_parameters(self.model)
        print(f"Model parameters - Trainable: {trainable:,}, Total: {total:,}")
        
    def setup_training(self):
        """Setup loss, optimizer, and scheduler."""
        self.criterion = nn.MSELoss()
        
        # Separate learning rates for backbone and fusion head
        if self.args.freeze_backbone:
            # Only train fusion MLP
            params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            # Different LR for backbone vs head
            backbone_params = []
            head_params = []
            
            for name, param in self.model.named_parameters():
                if 'resnet_backbone' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
            
            params = [
                {'params': backbone_params, 'lr': self.args.learning_rate * 0.1},
                {'params': head_params, 'lr': self.args.learning_rate}
            ]
        
        self.optimizer = optim.AdamW(params, lr=self.args.learning_rate, 
                                     weight_decay=self.args.weight_decay)
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', 
                                           factor=0.5, patience=3)
        
    def normalize_batch_features(self, physics_features):
        """Normalize physics features batch."""
        mean = self.feature_mean.to(self.device)
        std = self.feature_std.to(self.device)
        return (physics_features - mean) / std
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, physics_features, labels in pbar:
            images = images.to(self.device)
            physics_features = self.normalize_batch_features(physics_features.to(self.device))
            labels = labels.to(self.device).unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images, physics_features)
            
            # Compute loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return epoch_loss / len(self.train_loader)
    
    def validate(self, loader):
        """Validate model."""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, physics_features, labels in loader:
                images = images.to(self.device)
                physics_features = self.normalize_batch_features(physics_features.to(self.device))
                labels = labels.to(self.device).unsqueeze(1)
                
                predictions = self.model(images, physics_features)
                
                loss = self.criterion(predictions, labels)
                val_loss += loss.item()
                
                all_preds.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate all predictions
        all_preds = np.concatenate(all_preds, axis=0).flatten()
        all_labels = np.concatenate(all_labels, axis=0).flatten()
        
        # Inverse log transform to get original scale
        all_preds = inverse_log_transform_pm25(all_preds)
        all_labels = inverse_log_transform_pm25(all_labels)
        
        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds)
        avg_loss = val_loss / len(loader)
        
        return avg_loss, metrics, all_preds, all_labels
    
    def train(self):
        """Main training loop."""
        print(f"\n=== Starting training for {self.args.epochs} epochs ===\n")
        
        for epoch in range(1, self.args.epochs + 1):
            print(f"Epoch {epoch}/{self.args.epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics, _, _ = self.validate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_maes.append(val_metrics['mae'])
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['mae'])
            
            # Print epoch summary
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print_metrics_summary(val_metrics, "  Val")
            
            # Save best model
            if val_metrics['mae'] < self.best_val_mae:
                self.best_val_mae = val_metrics['mae']
                self.patience_counter = 0
                
                # Save checkpoint
                save_model(
                    self.model,
                    f"{self.args.checkpoint_dir}/best_model.pt",
                    metadata={
                        'epoch': epoch,
                        'val_mae': val_metrics['mae'],
                        'val_rmse': val_metrics['rmse'],
                        'val_spearman': val_metrics['spearman'],
                        'num_physics_features': self.num_physics_features,
                        'feature_names': self.feature_names
                    }
                )
                print(f"  ✓ New best model saved (MAE: {val_metrics['mae']:.2f})")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
            
            print()
        
        print("Training completed!")
        
    def evaluate_test(self):
        """Evaluate on test set."""
        print("\n=== Evaluating on test set ===")
        
        # Load best model
        checkpoint = torch.load(f"{self.args.checkpoint_dir}/best_model.pt", 
                               map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        test_loss, test_metrics, test_preds, test_labels = self.validate(self.test_loader)
        
        print_metrics_summary(test_metrics, "Test")
        
        # Save predictions
        results_df = pd.DataFrame({
            'true_pm25': test_labels,
            'pred_pm25': test_preds,
            'error': test_preds - test_labels
        })
        results_df.to_csv(f"{self.args.output_dir}/test_predictions.csv", index=False)
        
        # Plot results
        self.plot_results(test_labels, test_preds)
        
        return test_metrics
    
    def plot_results(self, y_true, y_pred):
        """Plot training curves and prediction results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Training curves
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE curve
        axes[0, 1].plot(self.val_maes, label='Val MAE', color='orange')
        axes[0, 1].axhline(y=self.best_val_mae, color='r', linestyle='--', 
                          label=f'Best MAE: {self.best_val_mae:.2f}')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE (µg/m³)')
        axes[0, 1].set_title('Validation MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Scatter plot
        axes[1, 0].scatter(y_true, y_pred, alpha=0.5)
        axes[1, 0].plot([y_true.min(), y_true.max()], 
                       [y_true.min(), y_true.max()], 
                       'r--', lw=2)
        axes[1, 0].set_xlabel('True PM2.5 (µg/m³)')
        axes[1, 0].set_ylabel('Predicted PM2.5 (µg/m³)')
        axes[1, 0].set_title('Predicted vs True PM2.5')
        axes[1, 0].grid(True)
        
        # Residual histogram
        residuals = y_pred - y_true
        axes[1, 1].hist(residuals, bins=30, edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Residual (µg/m³)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Residual Distribution')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.args.output_dir}/training_results.png", dpi=150)
        print(f"Results plot saved to {self.args.output_dir}/training_results.png")
        plt.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Aero-Gauge PM2.5 estimation model")
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./dataset',
                       help='Path to dataset directory')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                       help='Directory for caching features')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for logs and results')
    parser.add_argument('--checkpoint_dir', type=str, default='./weights',
                       help='Directory to save model checkpoints')
    
    # Model
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                       help='Freeze ResNet backbone layers 1-3')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout probability')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    
    # System
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Print configuration
    print("=" * 60)
    print("Aero-Gauge Training")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)
    
    # Initialize trainer
    trainer = Trainer(args)
    
    # Train
    trainer.train()
    
    # Evaluate on test set
    test_metrics = trainer.evaluate_test()
    
    # Save final summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'best_val_mae': float(trainer.best_val_mae),
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'num_physics_features': trainer.num_physics_features,
        'feature_names': trainer.feature_names
    }
    
    with open(f"{args.output_dir}/training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Best validation MAE: {trainer.best_val_mae:.2f} µg/m³")
    print(f"Test MAE: {test_metrics['mae']:.2f} µg/m³")
    print(f"Test Spearman: {test_metrics['spearman']:.4f}")
    print(f"Model saved to: {args.checkpoint_dir}/best_model.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
