"""
Unit tests for physics feature extraction.

Run with: python -m pytest tests/test_features.py
or: python tests/test_features.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import unittest
from features import (
    dark_channel, estimate_atmospheric_light, estimate_transmission,
    compute_laplacian_variance, compute_contrast, compute_color_attenuation,
    extract_all_features, get_dark_channel_heatmap
)


class TestFeatureExtraction(unittest.TestCase):
    """Test suite for physics-based feature extraction."""
    
    def setUp(self):
        """Create test images."""
        # Clean image (low haze)
        self.clean_image = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        
        # Hazy image (high haze)
        self.hazy_image = np.full((224, 224, 3), 180, dtype=np.uint8)
        self.hazy_image += np.random.randint(-20, 20, (224, 224, 3), dtype=np.uint8)
        
        # Black image
        self.black_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # White image
        self.white_image = np.full((224, 224, 3), 255, dtype=np.uint8)
    
    def test_dark_channel_shape(self):
        """Test dark channel output shape."""
        dark = dark_channel(self.clean_image, window_size=15)
        self.assertEqual(dark.shape, (224, 224))
        self.assertEqual(dark.dtype, np.uint8)
    
    def test_dark_channel_range(self):
        """Test dark channel values are in valid range."""
        dark = dark_channel(self.clean_image)
        self.assertTrue(np.all(dark >= 0))
        self.assertTrue(np.all(dark <= 255))
    
    def test_dark_channel_black_image(self):
        """Test dark channel on black image."""
        dark = dark_channel(self.black_image)
        # Black image should have very low dark channel
        self.assertTrue(np.mean(dark) < 10)
    
    def test_dark_channel_white_image(self):
        """Test dark channel on white image."""
        dark = dark_channel(self.white_image)
        # White image should have high dark channel
        self.assertTrue(np.mean(dark) > 200)
    
    def test_dark_channel_hazy_vs_clean(self):
        """Test that hazy image has higher dark channel than clean."""
        dark_clean = dark_channel(self.clean_image)
        dark_hazy = dark_channel(self.hazy_image)
        
        # Hazy image should generally have higher dark channel
        self.assertGreater(np.mean(dark_hazy), np.mean(dark_clean))
    
    def test_atmospheric_light_shape(self):
        """Test atmospheric light output shape."""
        dark = dark_channel(self.clean_image)
        atm_light = estimate_atmospheric_light(self.clean_image, dark)
        
        self.assertEqual(atm_light.shape, (3,))
        self.assertEqual(atm_light.dtype, np.float32)
    
    def test_atmospheric_light_range(self):
        """Test atmospheric light values are valid."""
        dark = dark_channel(self.clean_image)
        atm_light = estimate_atmospheric_light(self.clean_image, dark)
        
        self.assertTrue(np.all(atm_light >= 0))
        self.assertTrue(np.all(atm_light <= 255))
    
    def test_transmission_shape(self):
        """Test transmission map output shape."""
        dark = dark_channel(self.clean_image)
        atm_light = estimate_atmospheric_light(self.clean_image, dark)
        transmission = estimate_transmission(self.clean_image, atm_light)
        
        self.assertEqual(transmission.shape, (224, 224))
    
    def test_transmission_range(self):
        """Test transmission values are in [0, 1]."""
        dark = dark_channel(self.clean_image)
        atm_light = estimate_atmospheric_light(self.clean_image, dark)
        transmission = estimate_transmission(self.clean_image, atm_light)
        
        self.assertTrue(np.all(transmission >= 0))
        self.assertTrue(np.all(transmission <= 1))
    
    def test_laplacian_variance_positive(self):
        """Test Laplacian variance is positive."""
        lap_var = compute_laplacian_variance(self.clean_image)
        
        self.assertIsInstance(lap_var, float)
        self.assertGreater(lap_var, 0)
    
    def test_laplacian_variance_blurry_vs_sharp(self):
        """Test that blurry image has lower Laplacian variance."""
        # Create sharp image with edges
        sharp_image = np.zeros((224, 224, 3), dtype=np.uint8)
        sharp_image[:, :112] = 255
        
        # Create blurry image (uniform)
        blurry_image = np.full((224, 224, 3), 128, dtype=np.uint8)
        
        lap_sharp = compute_laplacian_variance(sharp_image)
        lap_blurry = compute_laplacian_variance(blurry_image)
        
        self.assertGreater(lap_sharp, lap_blurry)
    
    def test_contrast_positive(self):
        """Test contrast is positive."""
        contrast = compute_contrast(self.clean_image)
        
        self.assertIsInstance(contrast, float)
        self.assertGreater(contrast, 0)
    
    def test_contrast_uniform_vs_varied(self):
        """Test that uniform image has lower contrast."""
        uniform_image = np.full((224, 224, 3), 128, dtype=np.uint8)
        varied_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        contrast_uniform = compute_contrast(uniform_image)
        contrast_varied = compute_contrast(varied_image)
        
        self.assertGreater(contrast_varied, contrast_uniform)
    
    def test_color_attenuation_range(self):
        """Test color attenuation is in reasonable range."""
        attenuation = compute_color_attenuation(self.clean_image)
        
        self.assertIsInstance(attenuation, float)
        self.assertGreaterEqual(attenuation, 0)
        self.assertLessEqual(attenuation, 1)
    
    def test_extract_all_features_keys(self):
        """Test that extract_all_features returns expected keys."""
        features = extract_all_features(self.clean_image)
        
        # Check dictionary
        self.assertIsInstance(features, dict)
        
        # Check for key feature groups
        dark_channel_keys = [k for k in features.keys() if 'dark_channel' in k]
        transmission_keys = [k for k in features.keys() if 'transmission' in k]
        atmospheric_keys = [k for k in features.keys() if 'atmospheric' in k]
        
        self.assertGreater(len(dark_channel_keys), 0)
        self.assertGreater(len(transmission_keys), 0)
        self.assertGreater(len(atmospheric_keys), 0)
        
        # Check specific keys
        self.assertIn('dark_channel_mean', features)
        self.assertIn('laplacian_variance', features)
        self.assertIn('contrast', features)
        self.assertIn('color_attenuation', features)
    
    def test_extract_all_features_values(self):
        """Test that extracted features have valid values."""
        features = extract_all_features(self.clean_image)
        
        for key, value in features.items():
            # All features should be numeric
            self.assertIsInstance(value, (int, float, np.number))
            
            # No NaN or Inf
            self.assertFalse(np.isnan(value), f"{key} is NaN")
            self.assertFalse(np.isinf(value), f"{key} is Inf")
    
    def test_heatmap_shape(self):
        """Test dark channel heatmap output shape."""
        heatmap = get_dark_channel_heatmap(self.clean_image)
        
        self.assertEqual(heatmap.shape, (224, 224, 3))
        self.assertEqual(heatmap.dtype, np.uint8)
    
    def test_heatmap_range(self):
        """Test heatmap values are in valid range."""
        heatmap = get_dark_channel_heatmap(self.clean_image)
        
        self.assertTrue(np.all(heatmap >= 0))
        self.assertTrue(np.all(heatmap <= 255))
    
    def test_feature_extraction_different_sizes(self):
        """Test feature extraction works with different image sizes."""
        sizes = [(100, 100, 3), (224, 224, 3), (500, 500, 3)]
        
        for size in sizes:
            img = np.random.randint(0, 255, size, dtype=np.uint8)
            features = extract_all_features(img)
            
            self.assertIsInstance(features, dict)
            self.assertGreater(len(features), 0)
    
    def test_feature_reproducibility(self):
        """Test that same image produces same features."""
        features1 = extract_all_features(self.clean_image)
        features2 = extract_all_features(self.clean_image)
        
        for key in features1.keys():
            self.assertAlmostEqual(features1[key], features2[key], places=5,
                                 msg=f"Feature {key} not reproducible")


class TestFeatureIntegration(unittest.TestCase):
    """Integration tests for feature pipeline."""
    
    def test_full_pipeline(self):
        """Test full feature extraction pipeline."""
        # Create test image
        img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        
        # Extract all features
        features = extract_all_features(img)
        
        # Verify we have enough features
        self.assertGreaterEqual(len(features), 20)
        
        # Verify all values are numeric and valid
        for key, value in features.items():
            self.assertIsInstance(value, (int, float, np.number))
            self.assertFalse(np.isnan(value))
            self.assertFalse(np.isinf(value))
    
    def test_edge_cases(self):
        """Test edge cases."""
        # All zeros
        zero_img = np.zeros((224, 224, 3), dtype=np.uint8)
        features_zero = extract_all_features(zero_img)
        self.assertIsInstance(features_zero, dict)
        
        # All ones
        one_img = np.ones((224, 224, 3), dtype=np.uint8)
        features_one = extract_all_features(one_img)
        self.assertIsInstance(features_one, dict)
        
        # Random noise
        noise_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        features_noise = extract_all_features(noise_img)
        self.assertIsInstance(features_noise, dict)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
