"""
Physics-based feature extraction for air quality estimation.

This module implements Dark Channel Prior (DCP) and other handcrafted features
that correlate with atmospheric haze and PM2.5 levels.
"""

import cv2
import numpy as np
from typing import Tuple, Dict


def dark_channel(image: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    Compute the Dark Channel Prior of an image.
    
    The dark channel is the minimum intensity across RGB channels in local patches.
    In haze-free outdoor images, at least one color channel has very low intensity
    in most non-sky patches. Haze increases dark channel values.
    
    Args:
        image: RGB image as numpy array (H, W, 3), values in [0, 255]
        window_size: Size of local patch for minimum filter (default: 15)
        
    Returns:
        dark_channel_map: 2D array (H, W) with dark channel values
        
    Reference:
        He et al., "Single Image Haze Removal Using Dark Channel Prior", CVPR 2009
    """
    # Ensure image is in correct format
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Take minimum across RGB channels for each pixel
    min_channel = np.min(image, axis=2)
    
    # Apply minimum filter over local patches
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark = cv2.erode(min_channel, kernel)
    
    return dark


def estimate_atmospheric_light(image: np.ndarray, dark_channel_map: np.ndarray, 
                                top_percent: float = 0.001) -> np.ndarray:
    """
    Estimate atmospheric light from the brightest pixels in the dark channel.
    
    Atmospheric light A is estimated by selecting the top 0.1% brightest pixels
    in the dark channel, then taking the pixel with highest intensity in the
    original image among those candidates.
    
    Args:
        image: RGB image (H, W, 3)
        dark_channel_map: Dark channel (H, W)
        top_percent: Fraction of brightest pixels to consider (default: 0.001)
        
    Returns:
        atmospheric_light: RGB vector (3,) representing A
    """
    h, w = dark_channel_map.shape
    num_pixels = h * w
    num_brightest = max(int(num_pixels * top_percent), 1)
    
    # Flatten and find indices of brightest pixels in dark channel
    dark_flat = dark_channel_map.flatten()
    indices = np.argsort(dark_flat)[-num_brightest:]
    
    # Convert flat indices to 2D coordinates
    y_coords = indices // w
    x_coords = indices % w
    
    # Find pixel with maximum intensity among candidates
    max_intensity = 0
    atmospheric_light = np.zeros(3)
    
    for y, x in zip(y_coords, x_coords):
        intensity = np.sum(image[y, x])
        if intensity > max_intensity:
            max_intensity = intensity
            atmospheric_light = image[y, x].astype(np.float32)
    
    return atmospheric_light


def estimate_transmission(image: np.ndarray, atmospheric_light: np.ndarray, 
                         window_size: int = 15, omega: float = 0.95) -> np.ndarray:
    """
    Estimate transmission map using Dark Channel Prior.
    
    The transmission t(x) represents the portion of light that reaches the camera
    without being scattered. Lower transmission indicates more haze.
    
    Formula: t(x) = 1 - omega * dark_channel(I(x) / A)
    
    Args:
        image: RGB image (H, W, 3)
        atmospheric_light: Atmospheric light RGB vector (3,)
        window_size: Size for dark channel computation
        omega: Haze retention parameter, typically 0.95 (keeps some natural haze)
        
    Returns:
        transmission_map: Transmission values (H, W) in [0, 1]
    """
    # Normalize image by atmospheric light
    normalized = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        normalized[:, :, i] = image[:, :, i] / max(atmospheric_light[i], 1e-6)
    
    normalized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    
    # Compute dark channel of normalized image
    dark = dark_channel(normalized, window_size)
    
    # Estimate transmission
    transmission = 1.0 - omega * (dark.astype(np.float32) / 255.0)
    
    return transmission


def compute_laplacian_variance(image: np.ndarray) -> float:
    """
    Compute variance of Laplacian as a measure of image blurriness.
    
    Hazy images tend to have lower edge sharpness and thus lower Laplacian variance.
    
    Args:
        image: RGB or grayscale image
        
    Returns:
        laplacian_variance: Float value indicating sharpness (higher = sharper)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Compute Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Return variance
    return float(np.var(laplacian))


def compute_contrast(image: np.ndarray) -> float:
    """
    Compute global contrast as std/mean ratio.
    
    Hazy images have reduced contrast due to scattering.
    
    Args:
        image: RGB or grayscale image
        
    Returns:
        contrast: Float value (std/mean)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    contrast = std_val / max(mean_val, 1e-6)
    return float(contrast)


def compute_color_attenuation(image: np.ndarray) -> float:
    """
    Compute color attenuation coefficient.
    
    Haze preferentially scatters shorter wavelengths (blue) more than longer ones (red).
    This creates a characteristic color shift in hazy images.
    
    Args:
        image: RGB image (H, W, 3)
        
    Returns:
        color_attenuation: Measure of blue-red channel difference
    """
    # Separate channels
    r_channel = image[:, :, 0].astype(np.float32)
    g_channel = image[:, :, 1].astype(np.float32)
    b_channel = image[:, :, 2].astype(np.float32)
    
    # Compute brightness
    brightness = (r_channel + g_channel + b_channel) / 3.0
    
    # Compute saturation
    max_rgb = np.maximum(np.maximum(r_channel, g_channel), b_channel)
    min_rgb = np.minimum(np.minimum(r_channel, g_channel), b_channel)
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
    
    # Color attenuation is inversely related to saturation in bright areas
    bright_mask = brightness > 128
    if np.sum(bright_mask) > 0:
        attenuation = float(np.mean(1.0 - saturation[bright_mask]))
    else:
        attenuation = float(np.mean(1.0 - saturation))
    
    return attenuation


def compute_rms_contrast(image: np.ndarray) -> float:
    """
    Compute RMS (root mean square) contrast.
    
    Args:
        image: RGB or grayscale image
        
    Returns:
        rms_contrast: Float value
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    mean_val = np.mean(gray)
    rms = np.sqrt(np.mean((gray - mean_val) ** 2))
    
    return float(rms)


def extract_all_features(image: np.ndarray, window_size: int = 15) -> Dict[str, float]:
    """
    Extract all physics-based features from an image.
    
    Args:
        image: RGB image as numpy array (H, W, 3), values in [0, 255]
        window_size: Size for dark channel computation
        
    Returns:
        features_dict: Dictionary with all computed features
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    features = {}
    
    # Dark Channel Prior features
    dark = dark_channel(image, window_size)
    features['dark_channel_mean'] = float(np.mean(dark))
    features['dark_channel_median'] = float(np.median(dark))
    features['dark_channel_std'] = float(np.std(dark))
    features['dark_channel_max'] = float(np.max(dark))
    features['dark_channel_p75'] = float(np.percentile(dark, 75))
    features['dark_channel_p90'] = float(np.percentile(dark, 90))
    
    # Atmospheric light
    atm_light = estimate_atmospheric_light(image, dark)
    features['atmospheric_light_r'] = float(atm_light[0])
    features['atmospheric_light_g'] = float(atm_light[1])
    features['atmospheric_light_b'] = float(atm_light[2])
    features['atmospheric_light_mean'] = float(np.mean(atm_light))
    
    # Transmission map
    transmission = estimate_transmission(image, atm_light, window_size)
    features['transmission_mean'] = float(np.mean(transmission))
    features['transmission_median'] = float(np.median(transmission))
    features['transmission_min'] = float(np.min(transmission))
    features['transmission_p10'] = float(np.percentile(transmission, 10))
    
    # Sharpness and contrast
    features['laplacian_variance'] = compute_laplacian_variance(image)
    features['contrast'] = compute_contrast(image)
    features['rms_contrast'] = compute_rms_contrast(image)
    
    # Color features
    features['color_attenuation'] = compute_color_attenuation(image)
    
    # Additional simple features
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features['mean_intensity'] = float(np.mean(gray))
    features['std_intensity'] = float(np.std(gray))
    
    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    features['edge_density'] = float(np.sum(edges > 0) / edges.size)
    
    return features


def get_dark_channel_heatmap(image: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    Generate a color heatmap visualization of the dark channel.
    
    Args:
        image: RGB image (H, W, 3)
        window_size: Size for dark channel computation
        
    Returns:
        heatmap: RGB heatmap image (H, W, 3)
    """
    dark = dark_channel(image, window_size)
    
    # Normalize to 0-255
    dark_norm = cv2.normalize(dark, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(dark_norm, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return heatmap


if __name__ == "__main__":
    # Test feature extraction
    import os
    
    # Create a synthetic hazy image for testing
    test_image = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
    
    print("Testing feature extraction...")
    features = extract_all_features(test_image)
    
    print("\nExtracted features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nDark channel heatmap shape:", get_dark_channel_heatmap(test_image).shape)
    print("Feature extraction test completed successfully!")
