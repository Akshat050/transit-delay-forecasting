#!/usr/bin/env python3
"""
DelaySenseAI Streamlit Dashboard
Interactive web application for transit delay analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import os # Added for file handling

# Import our custom modules
from data_processor import GTFSDataProcessor
from delay_predictor import DelayPredictor
from visualizer import DelayVisualizer

def main():
    """Main dashboard function"""
    
    # Page configuration
    st.set_page_config(
        page_title="DelaySenseAI - Transit Delay Prediction",
        page_icon="🚌",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🚌 DelaySenseAI - Transit Delay Prediction System")
    st.markdown("---")
    
    # Sidebar for model selection
    st.sidebar.header("🔧 Model Configuration")
    
    # Option to load existing model
    model_option = st.sidebar.radio(
        "Choose option:",
        ["Load Existing Model", "Train New Model"],
        help="Load a previously trained model or train a new one"
    )
    
    if model_option == "Load Existing Model":
        # Load existing model
        model_files = []
        if os.path.exists('outputs'):
            model_files = [f for f in os.listdir('outputs') if f.endswith('.pkl') and 'delaysense_model' in f]
        
        if model_files:
            selected_model = st.sidebar.selectbox(
                "Select model file:",
                options=sorted(model_files),
                help="Choose a trained model to load"
            )
            
            if st.sidebar.button("Load Model"):
                try:
                    predictor = DelayPredictor()
                    model_path = os.path.join('outputs', selected_model)
                    predictor.load_model(model_path)
                    
                    # Load corresponding data
                    processor = GTFSDataProcessor(".")
                    processor.load_gtfs_data()
                    processor.preprocess_stop_times()
                    processor.merge_trip_data()
                    processor.calculate_trip_features()
                    processor.create_delay_simulation()
                    processor.create_features()
                    
                    # Load transfer risk data if available
                    transfer_risk_path = os.path.join('outputs', 'transfer_risk.csv')
                    if os.path.exists(transfer_risk_path):
                        transfer_risk = pd.read_csv(transfer_risk_path)
                    else:
                        transfer_risk = None
                    
                    st.success(f"✅ Model loaded successfully from {selected_model}")
                    st.session_state.model_loaded = True
                    st.session_state.predictor = predictor
                    st.session_state.processor = processor
                    st.session_state.transfer_risk = transfer_risk
                    
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
                    st.session_state.model_loaded = False
        else:
            st.sidebar.warning("No saved models found. Train a new model first.")
            st.session_state.model_loaded = False
    
    # Risk threshold
    risk_threshold = st.sidebar.slider(
        "High Risk Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.7,
        step=0.05,
        help="Threshold for classifying trips as high risk"
    )
    
    # Check if model is loaded
    if not st.session_state.get('model_loaded', False):
        # Show welcome screen
        st.info("👋 Welcome to DelaySenseAI!")
        st.write("""
        This dashboard provides comprehensive analysis of transit delay predictions using GTFS data.
        
        **To get started:**
        1. Use the sidebar to load a previously trained model, or
        2. Train a new model using the sidebar option
        3. Explore the interactive visualizations and analysis tools
        
        **Features:**
        - 🗺️ Interactive delay hotspot maps
        - 📈 Route-specific delay analysis
        - ⏰ Time-based delay patterns
        - 🤖 Model performance metrics
        - 📊 Detailed data tables
        """)
        
        # Show available models
        if os.path.exists('outputs'):
            model_files = [f for f in os.listdir('outputs') if f.endswith('.pkl') and 'delaysense_model' in f]
            if model_files:
                st.subheader("📁 Available Trained Models")
                st.write("Found the following trained models:")
                for model_file in sorted(model_files):
                    st.write(f"   - {model_file}")
                st.info("💡 Use the sidebar to load one of these models!")
        
        # Show sample data info
        st.subheader("📁 Available GTFS Files")
        try:
            files = [f for f in os.listdir(".") if f.endswith(".txt")]
            if files:
                st.write("Found the following GTFS files:")
                for file in sorted(files):
                    st.write(f"   - {file}")
            else:
                st.warning("No GTFS files found in the current directory")
        except Exception as e:
            st.error(f"Error reading directory: {str(e)}")
        
        return
    
    # Model is loaded, proceed with analysis
    predictor = st.session_state.predictor
    processor = st.session_state.processor
    transfer_risk = st.session_state.transfer_risk
    
    # Get predictions for visualization
    with st.spinner("Loading model predictions..."):
        try:
            result_data, high_risk_segments = predictor.identify_high_risk_segments(
                processor.stop_times, threshold=risk_threshold
            )
        except Exception as e:
            st.error(f"Error getting predictions: {str(e)}")
            return
    
    # Initialize visualizer
    visualizer = DelayVisualizer()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Hotspot Map", "📈 Route Analysis", "⏰ Time Analysis", 
        "🤖 Model Performance", "📊 Data Tables"
    ])
    
    with tab1:
        st.subheader("Delay Hotspot Map")
        st.write("Interactive map showing delay risk by stop location")
        
        # Create map
        delay_map = visualizer.create_delay_hotspot_map(
            result_data, processor.stops, risk_threshold
        )
        
        # Display map
        st.components.v1.html(delay_map._repr_html_(), height=600)
        
        # Show high-risk segments summary
        if len(high_risk_segments) > 0:
            st.subheader("High-Risk Segments Summary")
            st.write(f"Found {len(high_risk_segments)} high-risk segments")
            
            # Group by route
            route_risk = high_risk_segments.groupby('route_short_name')['delay_risk'].agg(['count', 'mean']).round(3)
            route_risk.columns = ['Segment Count', 'Avg Risk']
            route_risk = route_risk.sort_values('Avg Risk', ascending=False)
            
            st.dataframe(route_risk.head(20))
        else:
            st.warning("No high-risk segments found with current threshold")
    
    with tab2:
        st.subheader("Route-Specific Analysis")
        
        # Route selector (handle mixed data types properly)
        route_series = result_data['route_short_name'].astype(str)
        route_series = route_series[route_series.str.lower() != 'nan']
        routes = sorted(route_series.unique().tolist())
        
        selected_route = st.selectbox(
            "Select Route",
            options=routes,
            help="Choose a route to analyze in detail"
        )
        
        if selected_route:
            route_rows = result_data[result_data['route_short_name'].astype(str) == selected_route]
            if len(route_rows) == 0:
                st.warning("No data found for the selected route.")
            else:
                route_id = route_rows['route_id'].iloc[0]
                route_fig = visualizer.create_route_delay_analysis(result_data, route_id)
                st.plotly_chart(route_fig, use_container_width=True)
    
    with tab3:
        st.subheader("Time-Based Analysis")
        
        # Time distribution
        time_fig = visualizer.create_time_analysis(result_data)
        st.plotly_chart(time_fig, use_container_width=True)
        
        # Peak hour analysis
        try:
            peak_fig = visualizer.create_peak_hour_analysis(result_data)
            st.plotly_chart(peak_fig, use_container_width=True)
        except AttributeError:
            st.info("Peak hour analysis not available in this version")
    
    with tab4:
        st.subheader("Model Performance Dashboard")
        perf_fig = visualizer.create_model_performance_dashboard(predictor)
        st.plotly_chart(perf_fig, use_container_width=True)
        
        # Show calibrated metrics if available
        if predictor.metrics.get('auc_score_calibrated'):
            st.info(f"Calibrated AUC-ROC: {predictor.metrics['auc_score_calibrated']:.3f}")
        if predictor.metrics.get('best_threshold'):
            st.info(f"Suggested threshold (F1‑max): {predictor.metrics['best_threshold']:.2f}")
        if predictor.metrics.get('brier_score'):
            st.info(f"Brier score: {predictor.metrics['brier_score']:.4f} (lower is better)")
        
        # Calibration curve
        if hasattr(predictor, 'calibration_curve_data') and predictor.calibration_curve_data:
            import plotly.graph_objects as go
            frac = predictor.calibration_curve_data['fraction_of_positives']
            meanp = predictor.calibration_curve_data['mean_predicted_value']
            cal_fig = go.Figure()
            cal_fig.add_trace(go.Scatter(x=meanp, y=frac, mode='lines+markers', name='Calibration'))
            cal_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Perfect', line=dict(dash='dash')))
            cal_fig.update_layout(title='Calibration Curve', xaxis_title='Mean predicted value', yaxis_title='Fraction of positives')
            st.plotly_chart(cal_fig, use_container_width=True)
    
    with tab5:
        st.subheader("Data Tables and Statistics")
        
        # Overall statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Route Statistics")
            route_stats = result_data.groupby('route_short_name').agg({
                'delay_risk': ['count', 'mean', 'std'],
                'total_stops': 'first',
                'total_distance': 'first'
            }).round(3)
            route_stats.columns = ['Trip Count', 'Avg Risk', 'Risk Std', 'Total Stops', 'Total Distance']
            route_stats = route_stats.sort_values('Avg Risk', ascending=False)
            
            st.dataframe(route_stats.head(20))
        
        with col2:
            st.subheader("Stop Statistics")
            stop_stats = result_data.groupby('stop_name').agg({
                'delay_risk': ['count', 'mean', 'std']
            }).round(3)
            stop_stats.columns = ['Visit Count', 'Avg Risk', 'Risk Std']
            stop_stats = stop_stats.sort_values('Avg Risk', ascending=False)
            
            st.dataframe(stop_stats.head(20))
        
        # Transfer risk if available
        if transfer_risk is not None:
            st.subheader("Transfer Risk Analysis")
            st.dataframe(transfer_risk.head(20))
            st.download_button(
                "Download transfer_risk.csv",
                data=transfer_risk.to_csv(index=False).encode('utf-8'),
                file_name='transfer_risk.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()
