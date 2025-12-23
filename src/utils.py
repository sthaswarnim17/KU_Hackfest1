"""
Utility functions for data processing, AQI mapping, metrics, and helpers.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from typing import Tuple, List, Dict
import json


# AQI Category definitions based on PM2.5 levels (µg/m³)
# Reference: US EPA AQI standards
AQI_CATEGORIES = [
    (0, 12.0, "Good", "#00E400", "Air quality is satisfactory, and air pollution poses little or no risk."),
    (12.1, 35.4, "Moderate", "#FFFF00", "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."),
    (35.5, 55.4, "Unhealthy for Sensitive Groups", "#FF7E00", "Members of sensitive groups may experience health effects. The general public is less likely to be affected."),
    (55.5, 150.4, "Unhealthy", "#FF0000", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."),
    (150.5, 250.4, "Very Unhealthy", "#8F3F97", "Health alert: The risk of health effects is increased for everyone."),
    (250.5, 500.0, "Hazardous", "#7E0023", "Health warning of emergency conditions: everyone is more likely to be affected."),
]


def pm25_to_aqi_category(pm25: float) -> Tuple[str, str, str]:
    """
    Convert PM2.5 concentration to AQI category, color, and health advice.
    
    Args:
        pm25: PM2.5 concentration in µg/m³
        
    Returns:
        (category_name, color_hex, health_advice)
    """
    for low, high, category, color, advice in AQI_CATEGORIES:
        if low <= pm25 <= high:
            return category, color, advice
    
    # If exceeds all ranges, return hazardous
    return "Hazardous", "#7E0023", "Health warning of emergency conditions: everyone is more likely to be affected."


def pm25_to_aqi_index(pm25: float) -> int:
    """
    Convert PM2.5 concentration to AQI index value (0-500).
    
    Uses the EPA breakpoint formula:
    AQI = [(I_high - I_low) / (C_high - C_low)] * (C - C_low) + I_low
    
    Args:
        pm25: PM2.5 concentration in µg/m³
        
    Returns:
        aqi_index: Integer AQI value
    """
    # AQI breakpoints (C_low, C_high, I_low, I_high)
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
            return int(round(aqi))
    
    # If exceeds all ranges, cap at 500
    return 500


def stratified_split(df: pd.DataFrame, pm25_column: str = 'pm2_5', 
                     train_ratio: float = 0.8, val_ratio: float = 0.1,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train/val/test sets with stratification by PM2.5 ranges.
    
    Args:
        df: DataFrame with PM2.5 labels
        pm25_column: Name of PM2.5 column
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
        random_state: Random seed
        
    Returns:
        (train_df, val_df, test_df)
    """
    # Create stratification bins
    df = df.copy()
    df['pm25_bin'] = pd.cut(df[pm25_column], 
                             bins=[0, 12, 35.5, 55.5, 150.5, 1000],
                             labels=['0-12', '12-35', '35-55', '55-150', '150+'])
    
    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_ratio,
        stratify=df['pm25_bin'],
        random_state=random_state
    )
    
    # Second split: val vs test
    val_size = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size,
        stratify=temp_df['pm25_bin'],
        random_state=random_state
    )
    
    # Remove temporary bin column
    train_df = train_df.drop('pm25_bin', axis=1)
    val_df = val_df.drop('pm25_bin', axis=1)
    test_df = test_df.drop('pm25_bin', axis=1)
    
    return train_df, val_df, test_df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        y_true: True PM2.5 values
        y_pred: Predicted PM2.5 values
        
    Returns:
        Dictionary with MAE, RMSE, and Spearman correlation
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Spearman correlation
    spearman_corr, _ = spearmanr(y_true, y_pred)
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'spearman': float(spearman_corr),
        'r2': float(r2)
    }


def log_transform_pm25(pm25: np.ndarray) -> np.ndarray:
    """
    Apply log transform to PM2.5 values to reduce skewness.
    
    Args:
        pm25: PM2.5 values
        
    Returns:
        log(1 + pm25)
    """
    return np.log1p(pm25)


def inverse_log_transform_pm25(log_pm25: np.ndarray) -> np.ndarray:
    """
    Inverse log transform to get original PM2.5 scale.
    
    Args:
        log_pm25: Log-transformed values
        
    Returns:
        Original PM2.5 values
    """
    return np.expm1(log_pm25)


def normalize_features(features: np.ndarray, mean: np.ndarray = None, 
                       std: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using z-score normalization.
    
    Args:
        features: Feature array (N, D)
        mean: Mean for normalization (if None, compute from features)
        std: Std for normalization (if None, compute from features)
        
    Returns:
        (normalized_features, mean, std)
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
        std[std < 1e-8] = 1.0  # Avoid division by zero
    
    normalized = (features - mean) / std
    
    return normalized, mean, std


def save_normalization_stats(mean: np.ndarray, std: np.ndarray, 
                             feature_names: List[str], filepath: str):
    """
    Save feature normalization statistics to JSON.
    
    Args:
        mean: Mean values
        std: Standard deviation values
        feature_names: List of feature names
        filepath: Path to save JSON file
    """
    stats = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'feature_names': feature_names
    }
    
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)


def load_normalization_stats(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load feature normalization statistics from JSON.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        (mean, std, feature_names)
    """
    with open(filepath, 'r') as f:
        stats = json.load(f)
    
    mean = np.array(stats['mean'])
    std = np.array(stats['std'])
    feature_names = stats['feature_names']
    
    return mean, std, feature_names


def create_confidence_interval(predictions: List[np.ndarray], 
                               confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create confidence intervals from multiple model predictions (e.g., k-fold ensemble).
    
    Args:
        predictions: List of prediction arrays from different models
        confidence: Confidence level (default 0.95)
        
    Returns:
        (lower_bound, upper_bound)
    """
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    # Use z-score for confidence interval
    z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    
    lower = mean_pred - z_score * std_pred
    upper = mean_pred + z_score * std_pred
    
    return lower, upper


def print_metrics_summary(metrics: Dict[str, float], prefix: str = ""):
    """
    Pretty print metrics summary.
    
    Args:
        metrics: Dictionary of metric values
        prefix: Prefix for output (e.g., "Train", "Val", "Test")
    """
    print(f"\n{prefix} Metrics:")
    print(f"  MAE:      {metrics['mae']:.2f} µg/m³")
    print(f"  RMSE:     {metrics['rmse']:.2f} µg/m³")
    print(f"  R²:       {metrics['r2']:.4f}")
    print(f"  Spearman: {metrics['spearman']:.4f}")


def get_physics_baseline(features: np.ndarray, pm25: np.ndarray) -> Dict[str, float]:
    """
    Train a simple linear regression baseline using only physics features.
    
    This serves as a baseline to compare against the deep learning model.
    
    Args:
        features: Physics features array (N, D)
        pm25: PM2.5 labels (N,)
        
    Returns:
        Dictionary with baseline coefficients
    """
    from sklearn.linear_model import LinearRegression
    
    # Use log-transformed target
    log_pm25 = log_transform_pm25(pm25)
    
    # Train simple linear model
    model = LinearRegression()
    model.fit(features, log_pm25)
    
    # Make predictions
    log_pred = model.predict(features)
    pred = inverse_log_transform_pm25(log_pred)
    
    # Compute metrics
    metrics = compute_metrics(pm25, pred)
    
    print("\n=== Physics-Only Baseline (Linear Regression) ===")
    print_metrics_summary(metrics, "Baseline")
    
    return {
        'coefficients': model.coef_.tolist(),
        'intercept': float(model.intercept_),
        'metrics': metrics
    }


def format_feature_importance(feature_names: List[str], feature_values: np.ndarray,
                              top_n: int = 5) -> List[Tuple[str, float]]:
    """
    Format top N features by absolute value for display.
    
    Args:
        feature_names: List of feature names
        feature_values: Feature values
        top_n: Number of top features to return
        
    Returns:
        List of (feature_name, value) tuples
    """
    # Get absolute values and sort
    abs_values = np.abs(feature_values)
    top_indices = np.argsort(abs_values)[-top_n:][::-1]
    
    top_features = [(feature_names[i], float(feature_values[i])) 
                    for i in top_indices]
    
    return top_features


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test AQI mapping
    test_pm25_values = [8, 25, 45, 100, 200, 300]
    print("\nPM2.5 to AQI Category mapping:")
    for pm25 in test_pm25_values:
        category, color, advice = pm25_to_aqi_category(pm25)
        aqi = pm25_to_aqi_index(pm25)
        print(f"  PM2.5={pm25:.1f}: AQI={aqi}, Category={category}")
    
    # Test log transform
    pm25_array = np.array([10, 50, 100, 200])
    log_pm25 = log_transform_pm25(pm25_array)
    recovered = inverse_log_transform_pm25(log_pm25)
    print("\nLog transform test:")
    print(f"  Original: {pm25_array}")
    print(f"  Log:      {log_pm25}")
    print(f"  Recovered: {recovered}")
    
    # Test metrics
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([12, 19, 32, 38, 51])
    metrics = compute_metrics(y_true, y_pred)
    print("\nMetrics test:")
    print_metrics_summary(metrics, "Test")
    
    print("\nUtility tests completed successfully!")
