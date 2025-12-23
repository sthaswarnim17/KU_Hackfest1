"""
Pre-training verification - Check everything before starting training.
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path

def check_gpu():
    """Verify NVIDIA GPU is available and will be used."""
    print("=" * 60)
    print("GPU VERIFICATION")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        print("Training will use CPU (very slow)")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✓ CUDA Available: True")
    print(f"✓ GPU Device: {gpu_name}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    if "Intel" in gpu_name or "UHD" in gpu_name:
        print(f"❌ Wrong GPU detected: {gpu_name}")
        print("Expected: NVIDIA RTX 4050")
        return False
    
    if "NVIDIA" not in gpu_name:
        print(f"⚠ Warning: GPU may not be NVIDIA: {gpu_name}")
    
    # Test GPU computation
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        print("✓ GPU computation test passed")
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False
    
    print("\n✅ GPU verification passed!")
    return True


def check_dataset():
    """Verify dataset files exist and are valid."""
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    
    dataset_dir = Path("dataset")
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False
    
    # Check train/test directories
    for split in ['train', 'test']:
        split_dir = dataset_dir / split
        metadata_file = split_dir / "metadata.csv"
        images_dir = split_dir / "images"
        
        if not metadata_file.exists():
            print(f"❌ Missing: {metadata_file}")
            return False
        
        if not images_dir.exists():
            print(f"❌ Missing: {images_dir}")
            return False
        
        # Read metadata
        df = pd.read_csv(metadata_file)
        
        # Count actual image files
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        print(f"\n{split.upper()} SET:")
        print(f"  Metadata rows: {len(df)}")
        print(f"  Image files: {len(image_files)}")
        
        # Check for missing files
        missing_count = 0
        if 'filename' in df.columns:
            for filename in df['filename'].head(100):  # Check first 100
                img_path = images_dir / filename.strip()
                if not img_path.exists():
                    missing_count += 1
        
        if missing_count > 0:
            print(f"  ⚠ Warning: {missing_count}/100 files missing (will be filtered)")
        else:
            print(f"  ✓ Sample files verified")
    
    print("\n✅ Dataset verification passed!")
    return True


def check_code():
    """Verify all source files exist and can be imported."""
    print("\n" + "=" * 60)
    print("CODE VERIFICATION")
    print("=" * 60)
    
    sys.path.insert(0, 'src')
    
    modules = [
        'features',
        'utils',
        'dataset',
        'model',
        'train',
    ]
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}.py")
        except Exception as e:
            print(f"❌ {module_name}.py: {e}")
            return False
    
    print("\n✅ Code verification passed!")
    return True


def check_disk_space():
    """Check available disk space."""
    print("\n" + "=" * 60)
    print("DISK SPACE VERIFICATION")
    print("=" * 60)
    
    import shutil
    
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024 ** 3)
    
    print(f"Free space: {free_gb:.1f} GB")
    
    if free_gb < 2:
        print("⚠ Warning: Less than 2GB free space")
        print("Recommend at least 5GB for caching and model checkpoints")
    else:
        print("✓ Sufficient disk space")
    
    return True


def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("PRE-TRAINING VERIFICATION")
    print("Aero-Gauge PM2.5 Estimation")
    print("=" * 60)
    
    checks = [
        ("GPU", check_gpu),
        ("Dataset", check_dataset),
        ("Code", check_code),
        ("Disk Space", check_disk_space),
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
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "=" * 60)
        print("🎉 ALL CHECKS PASSED!")
        print("=" * 60)
        print("\nReady to train! Run:")
        print("  python src/train.py --data_dir ./dataset --epochs 50 --batch_size 32")
        print("\nExpected training time: 5-10 minutes on RTX 4050")
        print("GPU memory usage: ~2-3 GB")
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ SOME CHECKS FAILED")
        print("=" * 60)
        print("\nPlease fix the issues above before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
