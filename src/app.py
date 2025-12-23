"""
Streamlit app for Aero-Gauge PM2.5 estimation.

Interactive web interface for uploading images and getting PM2.5 predictions
with explainability visualizations.
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from inference import PM25Predictor, create_predictor, physics_only_baseline
from features import get_dark_channel_heatmap


# Page configuration
st.set_page_config(
    page_title="Aero-Gauge | PM2.5 Vision Estimator",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .feature-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        background-color: #f8f9fa;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load model predictor (cached)."""
    try:
        predictor = create_predictor(
            checkpoint_dir='./weights',
            output_dir='./outputs',
            device='cpu'
        )
        return predictor, None
    except FileNotFoundError as e:
        return None, str(e)


def get_aqi_gauge_html(aqi_value: float, aqi_category: str, aqi_color: str):
    """Generate HTML for AQI gauge visualization."""
    return f"""
    <div style="text-align: center; padding: 1rem;">
        <div style="position: relative; width: 200px; height: 200px; margin: 0 auto;">
            <svg viewBox="0 0 200 200">
                <!-- Background arc -->
                <circle cx="100" cy="100" r="80" fill="none" stroke="#e0e0e0" stroke-width="20"/>
                <!-- Colored arc -->
                <circle cx="100" cy="100" r="80" fill="none" stroke="{aqi_color}" stroke-width="20"
                        stroke-dasharray="{min(aqi_value/500 * 502, 502)} 502"
                        transform="rotate(-90 100 100)"/>
                <!-- Center text -->
                <text x="100" y="95" text-anchor="middle" font-size="32" font-weight="bold" fill="#333">
                    {int(aqi_value)}
                </text>
                <text x="100" y="120" text-anchor="middle" font-size="14" fill="#666">
                    AQI
                </text>
            </svg>
        </div>
        <div style="font-size: 1.5rem; font-weight: bold; color: {aqi_color}; margin-top: 0.5rem;">
            {aqi_category}
        </div>
    </div>
    """


def plot_dark_channel_comparison(original_img, heatmap_img):
    """Create side-by-side comparison of original and dark channel."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap_img)
    axes[1].set_title('Dark Channel Prior Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig


def format_feature_value(value: float) -> str:
    """Format feature value for display."""
    if abs(value) < 0.01:
        return f"{value:.4f}"
    elif abs(value) < 1:
        return f"{value:.3f}"
    else:
        return f"{value:.2f}"


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<div class="main-header">🌫️ Aero-Gauge</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Powered PM2.5 Estimation from Outdoor Images</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About Aero-Gauge")
        st.write("""
        **Aero-Gauge** estimates PM2.5 air pollution levels from outdoor images using:
        
        - 🧠 **Deep Learning**: ResNet18 vision backbone
        - ⚛️ **Physics-based Features**: Dark Channel Prior, atmospheric transmission
        - 🔍 **Explainability**: Visual heatmaps and feature analysis
        
        Developed by **Team ClimateStack**
        """)
        
        st.divider()
        
        st.header("Instructions")
        st.write("""
        1. Upload or capture an outdoor image
        2. Click "Analyze Image"
        3. View PM2.5 estimate and AQI category
        4. Explore explainability features
        """)
        
        st.divider()
        
        st.header("Settings")
        use_baseline = st.checkbox("Use Physics-Only Baseline", value=False,
                                   help="Use simple physics-based estimation without ML model")
        
        show_raw_features = st.checkbox("Show All Physics Features", value=False)
    
    # Load model
    predictor, error = load_predictor()
    
    if predictor is None and not use_baseline:
        st.error(f"⚠️ Model not found: {error}")
        st.info("""
        Please train the model first by running:
        ```bash
        python src/train.py --data_dir ./dataset --epochs 30
        ```
        
        Or enable "Use Physics-Only Baseline" in the sidebar.
        """)
        use_baseline = True
    
    # Main content
    st.header("📸 Upload Image")
    
    # Image input options
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    with col2:
        camera_photo = st.camera_input("Or take a photo")
    
    # Use either uploaded or camera image
    image_source = camera_photo if camera_photo is not None else uploaded_file
    
    if image_source is not None:
        # Load image
        image = Image.open(image_source).convert('RGB')
        
        # Display original image
        st.image(image, caption="Input Image", use_container_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            
            with st.spinner("Analyzing image and estimating PM2.5..."):
                
                if use_baseline or predictor is None:
                    # Use physics-only baseline
                    result = physics_only_baseline(image)
                    st.info("ℹ️ Using physics-only baseline estimation")
                else:
                    # Use ML model
                    result = predictor.predict(image)
                
                # Display results
                st.success("✅ Analysis complete!")
                
                # Main metrics
                st.header("📊 Results")
                
                col1, col2, col3 = st.columns([2, 2, 3])
                
                with col1:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric(
                        label="PM2.5 Concentration",
                        value=f"{result['pm25']:.1f} µg/m³"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric(
                        label="AQI Index",
                        value=f"{result['aqi_index']}"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    # AQI Gauge
                    gauge_html = get_aqi_gauge_html(
                        result['aqi_index'],
                        result['aqi_category'],
                        result['aqi_color']
                    )
                    st.markdown(gauge_html, unsafe_allow_html=True)
                
                # Health advice
                st.markdown(
                    f'<div class="warning-box"><strong>Health Advisory:</strong> {result["health_advice"]}</div>',
                    unsafe_allow_html=True
                )
                
                # Disclaimer
                st.caption("⚠️ This is a proxy estimate based on image analysis, not a certified air quality sensor.")
                
                # Explainability section
                st.header("🔬 Explainability")
                
                tab1, tab2, tab3 = st.tabs(["Dark Channel Prior", "Top Features", "All Features"])
                
                with tab1:
                    st.subheader("Dark Channel Prior Visualization")
                    st.write("""
                    The Dark Channel Prior (DCP) reveals atmospheric haze in the image. 
                    Brighter areas in the heatmap indicate higher haze concentration, 
                    which correlates with higher PM2.5 levels.
                    """)
                    
                    if 'dark_channel_heatmap' in result:
                        # Create comparison plot
                        fig = plot_dark_channel_comparison(
                            np.array(image),
                            result['dark_channel_heatmap']
                        )
                        st.pyplot(fig)
                    else:
                        # Generate heatmap for baseline
                        image_np = np.array(image.resize((224, 224)))
                        heatmap = get_dark_channel_heatmap(image_np)
                        fig = plot_dark_channel_comparison(
                            np.array(image.resize((224, 224))),
                            heatmap
                        )
                        st.pyplot(fig)
                
                with tab2:
                    st.subheader("Top Contributing Physics Features")
                    st.write("""
                    These are the physics-based features that most strongly influenced 
                    the PM2.5 prediction (sorted by importance).
                    """)
                    
                    if 'top_features' in result:
                        for i, feat in enumerate(result['top_features'], 1):
                            with st.container():
                                col1, col2, col3 = st.columns([3, 2, 2])
                                with col1:
                                    st.markdown(f"**{i}. {feat['name'].replace('_', ' ').title()}**")
                                with col2:
                                    st.text(f"Value: {format_feature_value(feat['raw_value'])}")
                                with col3:
                                    st.text(f"Z-score: {format_feature_value(feat['normalized_value'])}")
                    else:
                        st.info("Feature importance not available in baseline mode")
                
                with tab3:
                    if show_raw_features and 'physics_features' in result:
                        st.subheader("All Physics Features")
                        
                        features = result['physics_features']
                        
                        # Group features
                        dark_channel_features = {k: v for k, v in features.items() if 'dark_channel' in k}
                        transmission_features = {k: v for k, v in features.items() if 'transmission' in k}
                        atmospheric_features = {k: v for k, v in features.items() if 'atmospheric' in k}
                        other_features = {k: v for k, v in features.items() 
                                        if k not in dark_channel_features 
                                        and k not in transmission_features
                                        and k not in atmospheric_features}
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Dark Channel Features**")
                            for k, v in dark_channel_features.items():
                                st.text(f"{k}: {format_feature_value(v)}")
                            
                            st.markdown("**Transmission Features**")
                            for k, v in transmission_features.items():
                                st.text(f"{k}: {format_feature_value(v)}")
                        
                        with col2:
                            st.markdown("**Atmospheric Light**")
                            for k, v in atmospheric_features.items():
                                st.text(f"{k}: {format_feature_value(v)}")
                            
                            st.markdown("**Other Features**")
                            for k, v in other_features.items():
                                st.text(f"{k}: {format_feature_value(v)}")
                    else:
                        st.info("Enable 'Show All Physics Features' in the sidebar to view full feature set")
                
                # Download results
                st.header("💾 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # JSON export
                    export_data = {
                        'timestamp': datetime.now().isoformat(),
                        'pm25': result['pm25'],
                        'aqi_index': result['aqi_index'],
                        'aqi_category': result['aqi_category'],
                        'health_advice': result['health_advice']
                    }
                    
                    if 'physics_features' in result:
                        export_data['physics_features'] = result['physics_features']
                    
                    json_str = json.dumps(export_data, indent=2)
                    
                    st.download_button(
                        label="📄 Download Results (JSON)",
                        data=json_str,
                        file_name=f"aerogauge_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col2:
                    # Summary text export
                    summary_text = f"""
Aero-Gauge PM2.5 Estimation Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PM2.5 Concentration: {result['pm25']:.1f} µg/m³
AQI Index: {result['aqi_index']}
AQI Category: {result['aqi_category']}

Health Advisory:
{result['health_advice']}

Note: This is an AI-based estimate, not a certified measurement.
                    """
                    
                    st.download_button(
                        label="📝 Download Summary (TXT)",
                        data=summary_text,
                        file_name=f"aerogauge_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
    
    else:
        # Show example/demo state
        st.info("👆 Please upload an image or take a photo to begin")
        
        # Show sample images if available
        assets_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
        if os.path.exists(assets_dir):
            sample_images = [f for f in os.listdir(assets_dir) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if sample_images:
                st.subheader("Sample Images")
                cols = st.columns(min(3, len(sample_images)))
                for i, img_file in enumerate(sample_images[:3]):
                    with cols[i]:
                        img_path = os.path.join(assets_dir, img_file)
                        img = Image.open(img_path)
                        st.image(img, caption=img_file, use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>Aero-Gauge</strong> by Team ClimateStack</p>
        <p>Physics + ML Fusion for Air Quality Estimation</p>
        <p style="font-size: 0.8rem;">
            Powered by ResNet18, Dark Channel Prior, and PyTorch | 
            <a href="https://github.com/yourusername/aerogauge" target="_blank">GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
