# Demo Images Guide

This folder should contain 3 sample images for demonstration purposes:

## Required Demo Images

### 1. **clean_air.jpg**
- Description: Clear outdoor scene with good air quality
- Characteristics: High visibility, sharp details, good contrast
- Expected PM2.5: 0-12 µg/m³ (Good)
- Source: Clear day photo of landscape/cityscape

### 2. **moderate_haze.jpg**
- Description: Outdoor scene with moderate haze
- Characteristics: Slightly reduced visibility, some atmospheric haze
- Expected PM2.5: 35-55 µg/m³ (Unhealthy for Sensitive Groups)
- Source: Slightly hazy day photo

### 3. **heavy_pollution.jpg**
- Description: Outdoor scene with heavy pollution/smog
- Characteristics: Low visibility, white/gray atmospheric layer, low contrast
- Expected PM2.5: 150+ µg/m³ (Unhealthy/Very Unhealthy)
- Source: Heavily polluted city or smoggy day photo

---

## How to Add Demo Images

### Option 1: Use Your Own Photos
1. Take or find 3 outdoor photos matching the descriptions above
2. Resize to approximately 640x480 or similar (not critical)
3. Save as JPG with appropriate names
4. Place in this `assets/` folder

### Option 2: Use Dataset Images
```bash
# Copy sample images from the dataset
cp dataset/train/images/image_with_low_pm25.jpg assets/clean_air.jpg
cp dataset/train/images/image_with_medium_pm25.jpg assets/moderate_haze.jpg
cp dataset/train/images/image_with_high_pm25.jpg assets/heavy_pollution.jpg
```

### Option 3: Download Free Stock Images
Search for:
- "clear day landscape"
- "hazy city skyline"
- "smog pollution city"

Sites: Unsplash, Pexels, Pixabay (ensure license allows usage)

---

## Testing Demo Images

After adding images, test them:

```python
from PIL import Image
from src.inference import physics_only_baseline

# Test each image
for img_file in ['clean_air.jpg', 'moderate_haze.jpg', 'heavy_pollution.jpg']:
    img = Image.open(f'assets/{img_file}')
    result = physics_only_baseline(img)
    print(f"{img_file}: PM2.5 = {result['pm25']:.1f} µg/m³ ({result['aqi_category']})")
```

Expected order: clean_air < moderate_haze < heavy_pollution

---

## Current Status

**No demo images included in repository** to keep it lightweight.

**Action**: Add 3 demo images as described above before running the Streamlit app demo.

---

## Note

Demo images are optional. The Streamlit app works without them - users can upload their own images or use camera input.
