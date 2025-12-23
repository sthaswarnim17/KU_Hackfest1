# Aero-Gauge Quick Start Commands

## 🚀 Initial Setup (Run Once)

### 1. Install Dependencies
```bash
cd d:\Project\3rd
pip install -r requirements.txt
```

**Windows users with GPU:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Verify Dataset
```bash
# Check dataset structure
dir dataset\train\images
dir dataset\test\images

# View metadata
python -c "import pandas as pd; print(pd.read_csv('dataset/train/metadata.csv').head())"
```

---

## 🎯 Training Commands

### Quick Training (CPU, ~2-4 hours)
```bash
python src/train.py --data_dir ./dataset --epochs 20 --batch_size 8
```

### Full Training (GPU recommended, ~30 min)
```bash
python src/train.py --data_dir ./dataset --epochs 30 --batch_size 32
```

### Custom Training
```bash
python src/train.py \
    --data_dir ./dataset \
    --checkpoint_dir ./weights \
    --output_dir ./outputs \
    --batch_size 16 \
    --epochs 30 \
    --learning_rate 1e-4 \
    --patience 5 \
    --freeze_backbone
```

**Training Parameters:**
- `--batch_size`: 8 (low memory) | 16 (balanced) | 32 (high memory)
- `--epochs`: 20 (quick) | 30 (recommended) | 50 (thorough)
- `--learning_rate`: 1e-4 (stable) | 5e-5 (fine-tune)
- `--cpu`: Force CPU usage (no GPU)

---

## 📊 Evaluation Commands

### View Training Results
```bash
# Open results image
start outputs\training_results.png

# View training summary
python -c "import json; print(json.dumps(json.load(open('outputs/training_summary.json')), indent=2))"
```

### Analyze Test Predictions
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/test_predictions.csv')
print(df.describe())

# Plot predictions vs true
plt.scatter(df['true_pm25'], df['pred_pm25'])
plt.xlabel('True PM2.5')
plt.ylabel('Predicted PM2.5')
plt.title('Test Set Predictions')
plt.show()
```

---

## 🎨 Streamlit App Commands

### Run Locally
```bash
streamlit run src/app.py
```

App opens at: **http://localhost:8501**

### Run on Custom Port
```bash
streamlit run src/app.py --server.port 8080
```

### Run with Auto-Reload (Development)
```bash
streamlit run src/app.py --server.runOnSave true
```

---

## 🧪 Testing Commands

### Test Physics Features
```bash
python src/features.py
```

### Test Model Architecture
```bash
python src/model.py
```

### Run Unit Tests
```bash
python tests/test_features.py
```

### Test Inference
```python
from PIL import Image
from src.inference import create_predictor

predictor = create_predictor()
image = Image.open('assets/clean_air.jpg')
result = predictor.predict(image)
print(f"PM2.5: {result['pm25']:.1f} µg/m³")
```

---

## 📦 Deployment Commands

### Package Model for Deployment
```bash
# Create deployment package
mkdir deploy
cp weights/best_model.pt deploy/
cp outputs/feature_normalization.json deploy/
cp -r src deploy/
cp requirements.txt deploy/
cp README.md deploy/

# Zip for sharing
powershell Compress-Archive -Path deploy -DestinationPath aerogauge-v1.zip
```

### Deploy to Hugging Face Spaces

1. **Create Space:**
   - Go to https://huggingface.co/spaces
   - New Space → Select "Streamlit" SDK
   - Name: `your-username/aero-gauge`

2. **Prepare Files:**
```bash
# Create app.py in root (copy from src/app.py)
cp src/app.py app.py

# Modify imports in app.py (first line):
# Change: from inference import ...
# To: from src.inference import ...
```

3. **Push to Space:**
```bash
git init
git lfs install
git lfs track "*.pt"  # Track large model files
git add .
git commit -m "Initial commit"
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/aero-gauge
git push --force space main
```

4. **Add Model File:**
   - If model > 10MB, use Git LFS or upload via HF interface
   - Or host on Google Drive and download in app startup:
   
```python
# In app.py, before loading model:
import gdown
if not os.path.exists('weights/best_model.pt'):
    gdown.download('YOUR_GDRIVE_FILE_ID', 'weights/best_model.pt', quiet=False)
```

### Deploy to Streamlit Cloud

1. **Push to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/aero-gauge.git
git push -u origin main
```

2. **Deploy:**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Connect GitHub repository
   - Main file: `src/app.py`
   - Deploy!

**Note:** For large models, host weights externally and download on app startup.

---

## 🔄 Google Colab Training

### Upload to Colab
1. Open `notebooks/train_colab.ipynb` in Colab
2. Upload source files or mount Google Drive
3. Upload dataset or download via Kaggle API
4. Run all cells
5. Download trained model

### Download Kaggle Dataset in Colab
```python
# In Colab cell:
!pip install kaggle
# Upload kaggle.json credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d deadcardassian/pm25vision -p dataset/ --unzip
```

---

## 🐛 Troubleshooting Commands

### Check GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Clear Cache
```bash
# Remove cached features
rmdir /s cache

# Re-extract features
python -c "from src.dataset import PM25VisionDataset; ds = PM25VisionDataset('dataset/train/metadata.csv', 'dataset/train/images', cache_features=True)"
```

### Fix Import Errors
```bash
# Ensure you're in project root
cd d:\Project\3rd

# Test imports
python -c "import sys; sys.path.insert(0, 'src'); from features import extract_all_features; print('OK')"
```

### Memory Issues
```bash
# Reduce batch size
python src/train.py --batch_size 8

# Use CPU
python src/train.py --cpu

# Reduce workers
python src/train.py --num_workers 0
```

---

## 📈 Performance Tuning

### Increase Accuracy
```bash
# More epochs + lower learning rate
python src/train.py --epochs 50 --learning_rate 5e-5

# Unfreeze more layers (edit model.py freeze_backbone=False)
python src/train.py --freeze_backbone false
```

### Faster Training
```bash
# Increase batch size (needs more GPU memory)
python src/train.py --batch_size 64

# Reduce feature cache overhead
python src/train.py --num_workers 4
```

### Better Generalization
- Add more data augmentation (edit `src/dataset.py`)
- Increase dropout (e.g., `--dropout 0.4`)
- Use k-fold cross-validation

---

## 📋 Workflow Summary

```bash
# 1. Setup (once)
pip install -r requirements.txt

# 2. Train model
python src/train.py --data_dir ./dataset --epochs 30

# 3. Verify results
start outputs\training_results.png

# 4. Test inference
python src/inference.py

# 5. Launch app
streamlit run src/app.py

# 6. Deploy (optional)
# - Push to GitHub
# - Deploy to Streamlit Cloud or HF Spaces
```

---

## 🎯 One-Liner Commands

```bash
# Quick train + run
python src/train.py --epochs 20 && streamlit run src/app.py

# Train on GPU
python src/train.py --batch_size 32 --epochs 30

# Physics baseline only (no training needed)
streamlit run src/app.py
# (Enable "Physics-Only Baseline" in sidebar)
```

---

**Need help?** Check README.md for detailed documentation or open an issue on GitHub.
