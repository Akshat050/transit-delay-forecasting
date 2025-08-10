#!/usr/bin/env python3
"""
DelaySenseAI: Transit Delay Prediction System
Main execution script for the complete pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_processor import GTFSDataProcessor
from delay_predictor import DelayPredictor
from visualizer import DelayVisualizer

def main():
    """Main execution function for DelaySenseAI"""
    
    print("=" * 80)
    print("🚌 DelaySenseAI: Transit Delay Prediction System")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    DATA_DIR = "."  # Current directory containing GTFS files
    MODEL_TYPE = "xgboost"  # or "lightgbm"
    SAVE_MODEL = True
    CREATE_DASHBOARD = True
    OUTPUT_DIR = os.path.join('.', 'outputs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Step 1: Data Processing
        print("📊 Step 1: Processing GTFS Data")
        print("-" * 50)
        
        processor = GTFSDataProcessor(DATA_DIR)
        X, y = processor.process_all()
        
        print(f"✅ Data processing completed!")
        print(f"   - Features: {X.shape[1]}")
        print(f"   - Samples: {X.shape[0]}")
        print(f"   - Delay rate: {y.mean():.2%}")
        print()
        
        # Step 2: Model Training
        print("🤖 Step 2: Training Delay Prediction Model")
        print("-" * 50)
        
        predictor = DelayPredictor(model_type=MODEL_TYPE)
        predictor.train(X, y)
        
        print(f"✅ Model training completed!")
        print(f"   - Model type: {predictor.model_type.upper()}")
        print(f"   - Accuracy: {predictor.metrics['accuracy']:.2%}")
        print(f"   - AUC Score: {predictor.metrics['auc_score']:.3f}")
        print()
        
        # Step 3: High-Risk Segment Identification
        print("🎯 Step 3: Identifying High-Risk Segments")
        print("-" * 50)
        
        # Add delay risk predictions to the original data
        result_data, high_risk_segments = predictor.identify_high_risk_segments(
            processor.stop_times, threshold=0.7
        )
        
        print(f"✅ High-risk segment identification completed!")
        print(f"   - Total segments: {len(result_data)}")
        print(f"   - High-risk segments: {len(high_risk_segments)}")
        print(f"   - High-risk percentage: {len(high_risk_segments)/len(result_data):.1%}")
        print()

        # Step 3b: Transfer Reliability Analysis
        print("🔁 Step 3b: Computing Transfer Reliability")
        print("-" * 50)
        try:
            from transfer_processor import TransferDataBuilder
            t_builder = TransferDataBuilder(max_pairs=100000, min_window_min=2, max_window_min=20, alight_buffer_min=0)
            transfer_pairs = t_builder.build_pairs(processor.stop_times)
            if not transfer_pairs.empty:
                # Aggregate success by (stop, in_route, out_route)
                transfer_stats = (
                    transfer_pairs.groupby(['stop_id', 'in_route_id', 'out_route_id'])
                    .agg(success_rate=('transfer_success', 'mean'),
                         avg_wait_sched=('wait_minutes_sched', 'mean'),
                         n_pairs=('transfer_success', 'count'))
                    .reset_index()
                )
                transfer_stats['success_rate'] = transfer_stats['success_rate'].astype(float)
                transfer_stats = transfer_stats.sort_values('success_rate')
                transfer_stats.to_csv(os.path.join(OUTPUT_DIR, 'transfer_risk.csv'), index=False)
                print(f"✅ Transfer risk saved to transfer_risk.csv ({len(transfer_stats)} rows)")
                # Show top fragile transfers
                worst = transfer_stats.head(10)
                print("Top fragile transfers (lowest success rate):")
                for _, r in worst.iterrows():
                    print(f"   Stop {r['stop_id']}: {int(r['in_route_id'])} -> {int(r['out_route_id'])} | success={r['success_rate']:.1%} | n={int(r['n_pairs'])} | avg wait={r['avg_wait_sched']:.1f}m")
            else:
                print("No transfer pairs generated (check GTFS times window)")
        except Exception as e:
            print(f"⚠️ Transfer analysis skipped: {e}")
        print()
        
        # Step 4: Save Model (Optional)
        if SAVE_MODEL:
            print("💾 Step 4: Saving Trained Model")
            print("-" * 50)
            
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = os.path.join(OUTPUT_DIR, f"delaysense_model_{MODEL_TYPE}_{ts}.pkl")
            predictor.save_model(model_filename)
            # Save metrics
            import json
            # Convert numpy arrays to lists for JSON serialization
            metrics_for_json = {}
            for key, value in predictor.metrics.items():
                if isinstance(value, np.ndarray):
                    metrics_for_json[key] = value.tolist()
                else:
                    metrics_for_json[key] = value
            
            with open(os.path.join(OUTPUT_DIR, f"metrics_{ts}.json"), 'w', encoding='utf-8') as f:
                json.dump(metrics_for_json, f, indent=2)
            print(f"✅ Model saved as: {model_filename}")
            print()
        
        # Step 5: Create Visualizations
        print("📈 Step 5: Creating Visualizations")
        print("-" * 50)
        
        visualizer = DelayVisualizer()
        
        # Create delay hotspot map
        print("Creating delay hotspot map...")
        delay_map = visualizer.create_delay_hotspot_map(
            result_data, processor.stops, risk_threshold=0.7
        )
        delay_map.save(os.path.join(OUTPUT_DIR, "delay_hotspots.html"))
        print("✅ Delay hotspot map saved as: delay_hotspots.html")
        
        # Create route analysis (example with first route)
        print("Creating route analysis...")
        sample_route = result_data['route_id'].iloc[0]
        route_fig = visualizer.create_route_delay_analysis(result_data, sample_route)
        route_fig.write_html(os.path.join(OUTPUT_DIR, "route_analysis.html"))
        print("✅ Route analysis saved as: route_analysis.html")
        
        # Create time analysis
        print("Creating time-based analysis...")
        time_fig = visualizer.create_time_analysis(result_data)
        time_fig.write_html(os.path.join(OUTPUT_DIR, "time_analysis.html"))
        print("✅ Time analysis saved as: time_analysis.html")
        
        # Create model performance dashboard
        print("Creating model performance dashboard...")
        perf_fig = visualizer.create_model_performance_dashboard(predictor)
        perf_fig.write_html(os.path.join(OUTPUT_DIR, "model_performance.html"))
        print("✅ Model performance dashboard saved as: model_performance.html")
        
        print()
        
        # Step 6: Generate Summary Report
        print("📋 Step 6: Generating Summary Report")
        print("-" * 50)
        
        overall_stats, route_stats, stop_stats = visualizer.create_risk_summary_table(result_data)
        
        print("📊 OVERALL STATISTICS:")
        for key, value in overall_stats.items():
            print(f"   {key}: {value}")
        
        print("\n🚌 TOP 5 HIGH-RISK ROUTES:")
        top_routes = route_stats.head(5)
        for _, row in top_routes.iterrows():
            print(f"   {row['route_name']}: {row['avg_delay_risk']:.1%} risk ({row['trip_count']} trips)")
        
        print("\n🛑 TOP 5 HIGH-RISK STOPS:")
        top_stops = stop_stats.head(5)
        for _, row in top_stops.iterrows():
            print(f"   {row['stop_name']}: {row['avg_delay_risk']:.1%} risk ({row['trip_count']} trips)")
        
        print()
        
        # Step 7: Launch Dashboard (Optional)
        if CREATE_DASHBOARD:
            print("🌐 Step 7: Launching Streamlit Dashboard")
            print("-" * 50)
            print("To launch the interactive dashboard, run:")
            print("streamlit run dashboard.py")
            print()
        
        # Final Summary
        print("🎉 DELAYSENSEAI EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Generated Files (outputs/):")
        print("   - delay_hotspots.html (Interactive map)")
        print("   - route_analysis.html (Route-specific analysis)")
        print("   - time_analysis.html (Time-based analysis)")
        print("   - model_performance.html (Model metrics)")
        print("   - transfer_risk.csv (Transfer success probabilities)")
        if SAVE_MODEL:
            print(f"   - {model_filename} (Trained model)")
        print()
        print("Next Steps:")
        print("   1. Open the HTML files in a web browser to view visualizations")
        print("   2. Run 'streamlit run dashboard.py' for interactive dashboard")
        print("   3. Use the saved model for real-time predictions")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def create_dashboard_script():
    """Create the Streamlit dashboard script"""
    
    dashboard_code = '''#!/usr/bin/env python3
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

# Import our custom modules
from data_processor import GTFSDataProcessor
from delay_predictor import DelayPredictor
from visualizer import DelayVisualizer

def main():
    """Main dashboard function"""
    
    # Page configuration
    st.set_page_config(
        page_title="DelaySenseAI Dashboard",
        page_icon="🚌",
        layout="wide",
        initial_sidebar_state="expanded"
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
    st.markdown('<h1 class="main-header">🚌 DelaySenseAI</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Transit Delay Prediction System</h2>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["xgboost", "lightgbm"],
        help="Choose the machine learning model for delay prediction"
    )
    
    # Risk threshold
    risk_threshold = st.sidebar.slider(
        "High Risk Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.7,
        step=0.05,
        help="Threshold for classifying trips as high risk"
    )
    
    # Load or train model
    st.sidebar.subheader("🤖 Model Management")
    
    if st.sidebar.button("🔄 Process Data & Train Model"):
        with st.spinner("Processing GTFS data and training model..."):
            try:
                # Process data
                processor = GTFSDataProcessor(".")
                X, y = processor.process_all()
                
                # Train model
                predictor = DelayPredictor(model_type=model_type)
                predictor.train(X, y)
                
                # Store in session state
                st.session_state.processor = processor
                st.session_state.predictor = predictor
                st.session_state.X = X
                st.session_state.y = y
                
                st.success("✅ Model trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Load existing model
    uploaded_model = st.sidebar.file_uploader(
        "📁 Load Existing Model",
        type=['pkl'],
        help="Upload a previously trained model file"
    )
    
    if uploaded_model is not None:
        try:
            predictor = DelayPredictor()
            predictor.load_model(uploaded_model)
            st.session_state.predictor = predictor
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
    
    # Main content
    if 'predictor' in st.session_state and 'processor' in st.session_state:
        predictor = st.session_state.predictor
        processor = st.session_state.processor
        
        # Get processed data
        result_data, high_risk_segments = predictor.identify_high_risk_segments(
            processor.stop_times, threshold=risk_threshold
        )
        
        # Create visualizer
        visualizer = DelayVisualizer()
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Trips",
                len(result_data),
                help="Total number of transit trips analyzed"
            )
        
        with col2:
            avg_risk = result_data['delay_risk'].mean()
            st.metric(
                "Average Delay Risk",
                f"{avg_risk:.1%}",
                help="Average predicted delay risk across all trips"
            )
        
        with col3:
            high_risk_count = len(high_risk_segments)
            st.metric(
                "High Risk Trips",
                f"{high_risk_count} ({high_risk_count/len(result_data):.1%})",
                help="Number and percentage of trips with high delay risk"
            )
        
        with col4:
            st.metric(
                "Model Accuracy",
                f"{predictor.metrics['accuracy']:.1%}",
                help="Model prediction accuracy"
            )
        
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
            map_html = delay_map._repr_html_()
            st.components.v1.html(map_html, height=600)
        
        with tab2:
            st.subheader("Route-Specific Analysis")
            
            # Route selector
            routes = result_data['route_short_name'].unique()
            selected_route = st.selectbox(
                "Select Route",
                options=sorted(routes),
                help="Choose a route to analyze in detail"
            )
            
            if selected_route:
                route_id = result_data[result_data['route_short_name'] == selected_route]['route_id'].iloc[0]
                route_fig = visualizer.create_route_delay_analysis(result_data, route_id)
                st.plotly_chart(route_fig, use_container_width=True)
        
        with tab3:
            st.subheader("Time-Based Analysis")
            time_fig = visualizer.create_time_analysis(result_data)
            st.plotly_chart(time_fig, use_container_width=True)
        
        with tab4:
            st.subheader("Model Performance Dashboard")
            perf_fig = visualizer.create_model_performance_dashboard(predictor)
            st.plotly_chart(perf_fig, use_container_width=True)
        
        with tab5:
            st.subheader("Data Tables")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top High-Risk Routes**")
                overall_stats, route_stats, stop_stats = visualizer.create_risk_summary_table(result_data)
                top_routes = route_stats.head(10)
                st.dataframe(
                    top_routes[['route_name', 'avg_delay_risk', 'trip_count', 'high_risk_pct']],
                    use_container_width=True
                )
            
            with col2:
                st.write("**Top High-Risk Stops**")
                top_stops = stop_stats.head(10)
                st.dataframe(
                    top_stops[['stop_name', 'avg_delay_risk', 'trip_count', 'high_risk_pct']],
                    use_container_width=True
                )
    
    else:
        # Welcome screen
        st.info("👋 Welcome to DelaySenseAI!")
        st.write("""
        This dashboard provides comprehensive analysis of transit delay predictions using GTFS data.
        
        **To get started:**
        1. Use the sidebar to process your GTFS data and train a model
        2. Or upload a previously trained model file
        3. Explore the interactive visualizations and analysis tools
        
        **Features:**
        - 🗺️ Interactive delay hotspot maps
        - 📈 Route-specific delay analysis
        - ⏰ Time-based delay patterns
        - 🤖 Model performance metrics
        - 📊 Detailed data tables
        """)
        
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

if __name__ == "__main__":
    main()
'''
    
    with open("dashboard.py", "w", encoding="utf-8") as f:
        f.write(dashboard_code)
    
    print("✅ Created dashboard.py")

if __name__ == "__main__":
    # Create dashboard script
    create_dashboard_script()
    
    # Run main execution
    main() 