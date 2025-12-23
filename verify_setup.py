"""
Verification script to check all dependencies and basic functionality.
Run this before training to catch any issues early.
"""

import sys
import os

def check_imports():
    """Verify all required packages can be imported."""
    print("=" * 60)
    print("Checking package imports...")
    print("=" * 60)
    
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('streamlit', 'Streamlit'),
    ]
    
    failed = []
    for package, name in packages:
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name:15} {version}")
        except ImportError as e:
            print(f"✗ {name:15} MISSING")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt --user")
        return False
    
    print("\n✓ All packages imported successfully!")
    return True


def check_modules():
    """Verify all custom modules can be imported."""
    print("\n" + "=" * 60)
    print("Checking custom modules...")
    print("=" * 60)
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    modules = [
        'features',
        'utils',
        'dataset',
        'model',
        'train',
        'inference',
        'app',
    ]
    
    failed = []
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"✗ {module_name}: {e}")
            failed.append(module_name)
    
    if failed:
        print(f"\n❌ Failed modules: {', '.join(failed)}")
        return False
    
    print("\n✓ All modules imported successfully!")
    return True


def check_dataset():
    """Check if dataset structure is correct."""
    print("\n" + "=" * 60)
    print("Checking dataset structure...")
    print("=" * 60)
    
    required_paths = [
        'dataset/train/metadata.csv',
        'dataset/train/images',
        'dataset/test/metadata.csv',
        'dataset/test/images',
    ]
    
    missing = []
    for path in required_paths:
        full_path = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(full_path):
            if path.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(full_path)
                print(f"✓ {path} ({len(df)} rows)")
            else:
                count = len(os.listdir(full_path))
                print(f"✓ {path} ({count} files)")
        else:
            print(f"✗ {path} MISSING")
            missing.append(path)
    
    if missing:
        print(f"\n❌ Missing paths: {', '.join(missing)}")
        return False
    
    print("\n✓ Dataset structure is correct!")
    return True


def check_features():
    """Test feature extraction on a sample image."""
    print("\n" + "=" * 60)
    print("Testing feature extraction...")
    print("=" * 60)
    
    try:
        import numpy as np
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from features import extract_all_features
        
        # Create test image
        test_img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        
        # Extract features
        features = extract_all_features(test_img)
        
        print(f"✓ Extracted {len(features)} features")
        print(f"✓ Feature names: {', '.join(list(features.keys())[:5])}...")
        
        # Verify no NaN or inf
        feature_values = list(features.values())
        if any(np.isnan(feature_values)) or any(np.isinf(feature_values)):
            print("✗ Features contain NaN or inf values!")
            return False
        
        print("\n✓ Feature extraction works correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_pytorch():
    """Check PyTorch configuration."""
    print("\n" + "=" * 60)
    print("Checking PyTorch configuration...")
    print("=" * 60)
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Device count: {torch.cuda.device_count()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ No GPU detected - training will use CPU (slower)")
        
        # Test basic operations
        x = torch.randn(10, 10)
        y = torch.matmul(x, x)
        print(f"✓ PyTorch operations work correctly")
        
        # Test model loading
        from torchvision import models
        model = models.resnet18(weights='IMAGENET1K_V1')
        print(f"✓ Can load pretrained models")
        
        return True
        
    except Exception as e:
        print(f"✗ PyTorch check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("AERO-GAUGE SETUP VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("Package Imports", check_imports),
        ("Custom Modules", check_modules),
        ("Dataset Structure", check_dataset),
        ("Feature Extraction", check_features),
        ("PyTorch Configuration", check_pytorch),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} check crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "=" * 60)
        print("🎉 ALL CHECKS PASSED!")
        print("=" * 60)
        print("\nYou can now run training:")
        print("  python src/train.py --data_dir ./dataset --epochs 30 --batch_size 16")
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ SOME CHECKS FAILED")
        print("=" * 60)
        print("\nPlease fix the issues above before training.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
