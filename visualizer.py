import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import folium
from folium import plugins
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DelayVisualizer:
    """
    Visualization tools for transit delay analysis
    """
    
    def __init__(self):
        self.color_scheme = {
            'low_risk': '#00ff00',    # Green
            'medium_risk': '#ffff00', # Yellow
            'high_risk': '#ff0000',   # Red
            'very_high_risk': '#8b0000' # Dark Red
        }
        
    def create_delay_hotspot_map(self, stop_times_data, stops_data, risk_threshold=0.7):
        """Create an interactive map showing delay hotspots with cluster/heatmap overlays"""
        print("Creating delay hotspot map...")
        
        # Aggregate delay risk by stop
        stop_risk = stop_times_data.groupby('stop_id').agg({
            'delay_risk': ['mean', 'count'],
            'stop_name': 'first',
            'stop_lat': 'first',
            'stop_lon': 'first'
        }).reset_index()
        
        # Flatten column names
        stop_risk.columns = ['stop_id', 'avg_delay_risk', 'trip_count', 'stop_name', 'stop_lat', 'stop_lon']
        
        # Filter out stops with missing coordinates
        stop_risk = stop_risk.dropna(subset=['stop_lat', 'stop_lon'])
        
        # Create risk categories
        stop_risk['risk_category'] = pd.cut(
            stop_risk['avg_delay_risk'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
        )
        
        # Create the map
        center_lat = stop_risk['stop_lat'].mean()
        center_lon = stop_risk['stop_lon'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # Optional clustering
        cluster = plugins.MarkerCluster().add_to(m)

        # Add stops to map with color coding
        for idx, row in stop_risk.iterrows():
            # Determine color based on risk
            if row['avg_delay_risk'] < 0.3:
                color = self.color_scheme['low_risk']
            elif row['avg_delay_risk'] < 0.5:
                color = self.color_scheme['medium_risk']
            elif row['avg_delay_risk'] < 0.7:
                color = self.color_scheme['high_risk']
            else:
                color = self.color_scheme['very_high_risk']
            
            # Calculate radius based on trip count
            radius = min(20, max(5, row['trip_count'] / 100))
            
            # Create popup content
            popup_content = f"""
            <b>{row['stop_name']}</b><br>
            Stop ID: {row['stop_id']}<br>
            Average Delay Risk: {row['avg_delay_risk']:.2%}<br>
            Trip Count: {row['trip_count']}<br>
            Risk Category: {row['risk_category']}
            """
            
            folium.CircleMarker(
                location=[row['stop_lat'], row['stop_lon']],
                radius=radius,
                popup=folium.Popup(popup_content, max_width=300),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(cluster)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Delay Risk Legend</b></p>
        <p><span style="color:{};">●</span> Low Risk (&lt;30%)</p>
        <p><span style="color:{};">●</span> Medium Risk (30-50%)</p>
        <p><span style="color:{};">●</span> High Risk (50-70%)</p>
        <p><span style="color:{};">●</span> Very High Risk (&gt;70%)</p>
        </div>
        '''.format(
            self.color_scheme['low_risk'],
            self.color_scheme['medium_risk'],
            self.color_scheme['high_risk'],
            self.color_scheme['very_high_risk']
        )
        
        m.get_root().html.add_child(folium.Element(legend_html))

        # Heatmap layer
        try:
            heat = [[row['stop_lat'], row['stop_lon'], float(row['avg_delay_risk'])] for _, row in stop_risk.iterrows()]
            plugins.HeatMap(heat, radius=12, blur=15, name='Heatmap', min_opacity=0.4).add_to(m)
            folium.LayerControl().add_to(m)
        except Exception:
            pass
        
        return m
    
    @staticmethod
    def filter_by_hour(stop_times_data, hours):
        if hours is None:
            return stop_times_data
        return stop_times_data[stop_times_data['arrival_hour'].isin(hours)]
        
    def create_route_delay_analysis(self, stop_times_data, route_id=None):
        """Create route-specific delay analysis"""
        print("Creating route delay analysis...")
        
        # Filter by route if specified
        if route_id:
            route_data = stop_times_data[stop_times_data['route_id'] == route_id].copy()
            route_name = route_data['route_long_name'].iloc[0] if len(route_data) > 0 else f"Route {route_id}"
        else:
            route_data = stop_times_data.copy()
            route_name = "All Routes"
        
        # Aggregate by stop sequence for the route
        route_analysis = route_data.groupby(['stop_sequence', 'stop_name']).agg({
            'delay_risk': ['mean', 'std', 'count'],
            'stop_lat': 'first',
            'stop_lon': 'first'
        }).reset_index()
        
        # Flatten column names
        route_analysis.columns = ['stop_sequence', 'stop_name', 'avg_delay_risk', 'std_delay_risk', 'trip_count', 'stop_lat', 'stop_lon']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Delay Risk by Stop', 'Trip Count by Stop', 'Risk Distribution', 'Risk vs Trip Count'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Delay risk by stop
        fig.add_trace(
            go.Scatter(
                x=route_analysis['stop_sequence'],
                y=route_analysis['avg_delay_risk'],
                mode='lines+markers',
                name='Delay Risk',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Plot 2: Trip count by stop
        fig.add_trace(
            go.Bar(
                x=route_analysis['stop_sequence'],
                y=route_analysis['trip_count'],
                name='Trip Count',
                marker_color='blue'
            ),
            row=1, col=2
        )
        
        # Plot 3: Risk distribution
        fig.add_trace(
            go.Histogram(
                x=route_data['delay_risk'],
                nbinsx=30,
                name='Risk Distribution',
                marker_color='green'
            ),
            row=2, col=1
        )
        
        # Plot 4: Risk vs Trip Count scatter
        fig.add_trace(
            go.Scatter(
                x=route_analysis['trip_count'],
                y=route_analysis['avg_delay_risk'],
                mode='markers',
                name='Risk vs Trips',
                marker=dict(
                    size=10,
                    color=route_analysis['avg_delay_risk'],
                    colorscale='RdYlGn_r',
                    showscale=True
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Route Delay Analysis: {route_name}",
            height=800,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Stop Sequence", row=1, col=1)
        fig.update_yaxes(title_text="Average Delay Risk", row=1, col=1)
        fig.update_xaxes(title_text="Stop Sequence", row=1, col=2)
        fig.update_yaxes(title_text="Trip Count", row=1, col=2)
        fig.update_xaxes(title_text="Delay Risk", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Trip Count", row=2, col=2)
        fig.update_yaxes(title_text="Average Delay Risk", row=2, col=2)
        
        return fig
        
    def create_time_analysis(self, stop_times_data):
        """Create time-based delay analysis"""
        print("Creating time-based analysis...")
        
        # Create time-based aggregations
        hourly_risk = stop_times_data.groupby('arrival_hour').agg({
            'delay_risk': ['mean', 'count'],
            'is_peak_hour': 'first'
        }).reset_index()
        
        hourly_risk.columns = ['hour', 'avg_delay_risk', 'trip_count', 'is_peak_hour']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Delay Risk by Hour', 'Trip Volume by Hour', 'Peak vs Off-Peak Risk', 'Risk Distribution by Hour'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Delay risk by hour
        fig.add_trace(
            go.Scatter(
                x=hourly_risk['hour'],
                y=hourly_risk['avg_delay_risk'],
                mode='lines+markers',
                name='Delay Risk',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Plot 2: Trip count by hour
        fig.add_trace(
            go.Bar(
                x=hourly_risk['hour'],
                y=hourly_risk['trip_count'],
                name='Trip Count',
                marker_color='blue'
            ),
            row=1, col=2
        )
        
        # Plot 3: Peak vs Off-peak comparison
        peak_vs_offpeak = stop_times_data.groupby('is_peak_hour').agg({
            'delay_risk': ['mean', 'std', 'count']
        }).reset_index()
        
        peak_vs_offpeak.columns = ['is_peak_hour', 'avg_delay_risk', 'std_delay_risk', 'trip_count']
        
        fig.add_trace(
            go.Bar(
                x=['Off-Peak', 'Peak Hours'],
                y=peak_vs_offpeak['avg_delay_risk'],
                name='Peak vs Off-Peak',
                marker_color=['green', 'red']
            ),
            row=2, col=1
        )
        
        # Plot 4: Box plot by hour
        hourly_data = []
        hourly_labels = []
        
        for hour in range(24):
            hour_data = stop_times_data[stop_times_data['arrival_hour'] == hour]['delay_risk']
            if len(hour_data) > 0:
                hourly_data.append(hour_data.values)
                hourly_labels.append(f"{hour:02d}:00")
        
        fig.add_trace(
            go.Box(
                y=hourly_data,
                x=hourly_labels,
                name='Risk Distribution',
                boxpoints='outliers'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Time-Based Delay Analysis",
            height=800,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
        fig.update_yaxes(title_text="Average Delay Risk", row=1, col=1)
        fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
        fig.update_yaxes(title_text="Trip Count", row=1, col=2)
        fig.update_xaxes(title_text="Time Period", row=2, col=1)
        fig.update_yaxes(title_text="Average Delay Risk", row=2, col=1)
        fig.update_xaxes(title_text="Hour of Day", row=2, col=2)
        fig.update_yaxes(title_text="Delay Risk", row=2, col=2)
        
        return fig
        
    def create_model_performance_dashboard(self, predictor):
        """Create model performance dashboard"""
        print("Creating model performance dashboard...")
        
        if not predictor.is_trained:
            raise ValueError("Model must be trained first")
        
        # Get feature importance
        feature_importance = predictor.get_feature_importance(15)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Importance', 'Model Metrics', 'Confusion Matrix', 'Top Features Detail'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Feature importance
        fig.add_trace(
            go.Bar(
                x=feature_importance['importance'],
                y=feature_importance['feature'],
                orientation='h',
                name='Feature Importance',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Plot 2: Model metrics
        metrics = predictor.metrics
        fig.add_trace(
            go.Bar(
                x=['Accuracy', 'AUC Score'],
                y=[metrics['accuracy'], metrics['auc_score']],
                name='Model Performance',
                marker_color=['green', 'blue']
            ),
            row=1, col=2
        )
        
        # Plot 3: Confusion matrix heatmap
        cm = metrics['confusion_matrix']
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Predicted No Delay', 'Predicted Delay'],
                y=['Actual No Delay', 'Actual Delay'],
                colorscale='Blues',
                showscale=True
            ),
            row=2, col=1
        )
        
        # Plot 4: Top features with values
        top_features = feature_importance.head(10)
        fig.add_trace(
            go.Scatter(
                x=top_features['feature'],
                y=top_features['importance'],
                mode='markers+text',
                text=top_features['importance'].round(4),
                textposition='top center',
                name='Top Features',
                marker=dict(size=12, color='red')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Model Performance Dashboard - {predictor.model_type.upper()}",
            height=800,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Importance Score", row=1, col=1)
        fig.update_yaxes(title_text="Features", row=1, col=1)
        fig.update_xaxes(title_text="Metric", row=1, col=2)
        fig.update_yaxes(title_text="Score", row=1, col=2)
        fig.update_xaxes(title_text="Predicted", row=2, col=1)
        fig.update_yaxes(title_text="Actual", row=2, col=1)
        fig.update_xaxes(title_text="Features", row=2, col=2)
        fig.update_yaxes(title_text="Importance", row=2, col=2)
        
        return fig
        
    def create_risk_summary_table(self, stop_times_data):
        """Create a summary table of risk statistics"""
        print("Creating risk summary table...")
        
        # Overall statistics
        overall_stats = {
            'Total Trips': len(stop_times_data),
            'Average Delay Risk': f"{stop_times_data['delay_risk'].mean():.2%}",
            'High Risk Trips (>70%)': f"{(stop_times_data['delay_risk'] > 0.7).sum()} ({(stop_times_data['delay_risk'] > 0.7).mean():.1%})",
            'Medium Risk Trips (30-70%)': f"{((stop_times_data['delay_risk'] >= 0.3) & (stop_times_data['delay_risk'] <= 0.7)).sum()} ({((stop_times_data['delay_risk'] >= 0.3) & (stop_times_data['delay_risk'] <= 0.7)).mean():.1%})",
            'Low Risk Trips (<30%)': f"{(stop_times_data['delay_risk'] < 0.3).sum()} ({(stop_times_data['delay_risk'] < 0.3).mean():.1%})"
        }
        
        # Route-level statistics
        route_stats = stop_times_data.groupby(['route_id', 'route_short_name']).agg({
            'delay_risk': ['mean', 'count'],
            'high_risk': 'sum'
        }).reset_index()
        
        route_stats.columns = ['route_id', 'route_name', 'avg_delay_risk', 'trip_count', 'high_risk_count']
        route_stats['high_risk_pct'] = route_stats['high_risk_count'] / route_stats['trip_count']
        route_stats = route_stats.sort_values('avg_delay_risk', ascending=False)
        
        # Stop-level statistics
        stop_stats = stop_times_data.groupby(['stop_id', 'stop_name']).agg({
            'delay_risk': ['mean', 'count'],
            'high_risk': 'sum'
        }).reset_index()
        
        stop_stats.columns = ['stop_id', 'stop_name', 'avg_delay_risk', 'trip_count', 'high_risk_count']
        stop_stats['high_risk_pct'] = stop_stats['high_risk_count'] / stop_stats['trip_count']
        stop_stats = stop_stats.sort_values('avg_delay_risk', ascending=False)
        
        return overall_stats, route_stats, stop_stats
        
    def create_streamlit_dashboard(self, stop_times_data, predictor, stops_data):
        """Create a comprehensive Streamlit dashboard"""
        st.set_page_config(page_title="DelaySenseAI Dashboard", layout="wide")
        
        st.title("🚌 DelaySenseAI: Transit Delay Prediction System")
        st.markdown("---")
        
        # Sidebar
        st.sidebar.header("Dashboard Controls")
        
        # Risk threshold slider
        risk_threshold = st.sidebar.slider(
            "High Risk Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="Threshold for classifying trips as high risk"
        )
        
        # Route selector
        routes = stop_times_data['route_short_name'].unique()
        selected_route = st.sidebar.selectbox(
            "Select Route for Analysis",
            options=['All Routes'] + sorted(routes.tolist())
        )

        # Time window filter
        st.sidebar.subheader("Time Window Filter")
        hours_to_filter = st.sidebar.multiselect(
            "Select hours to include",
            options=range(24),
            default=None,
            help="Filter data by specific hours of the day"
        )
        if hours_to_filter:
            stop_times_data = self.filter_by_hour(stop_times_data, hours_to_filter)
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Risk Summary")
            overall_stats, route_stats, stop_stats = self.create_risk_summary_table(stop_times_data)
            
            # Display overall statistics
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            with stats_col1:
                st.metric("Total Trips", overall_stats['Total Trips'])
            with stats_col2:
                st.metric("Avg Delay Risk", overall_stats['Average Delay Risk'])
            with stats_col3:
                st.metric("High Risk Trips", overall_stats['High Risk Trips (>70%)'])
            with stats_col4:
                st.metric("Low Risk Trips", overall_stats['Low Risk Trips (<30%)'])
        
        with col2:
            st.subheader("🎯 Model Performance")
            if predictor.is_trained:
                st.metric("Accuracy", f"{predictor.metrics['accuracy']:.2%}")
                st.metric("AUC Score", f"{predictor.metrics['auc_score']:.3f}")
                st.metric("Model Type", predictor.model_type.upper())
            else:
                st.warning("Model not trained yet")
        
        # Maps and charts
        st.subheader("🗺️ Delay Hotspot Map")
        
        # Create map
        delay_map = self.create_delay_hotspot_map(stop_times_data, stops_data, risk_threshold)
        
        # Convert folium map to streamlit
        map_html = delay_map._repr_html_()
        st.components.v1.html(map_html, height=500)
        
        # Route analysis
        if selected_route != 'All Routes':
            st.subheader(f"📈 Route Analysis: {selected_route}")
            route_id = stop_times_data[stop_times_data['route_short_name'] == selected_route]['route_id'].iloc[0]
            route_fig = self.create_route_delay_analysis(stop_times_data, route_id)
            st.plotly_chart(route_fig, use_container_width=True)
        
        # Time analysis
        st.subheader("⏰ Time-Based Analysis")
        time_fig = self.create_time_analysis(stop_times_data)
        st.plotly_chart(time_fig, use_container_width=True)
        
        # Model performance
        if predictor.is_trained:
            st.subheader("🤖 Model Performance")
            perf_fig = self.create_model_performance_dashboard(predictor)
            st.plotly_chart(perf_fig, use_container_width=True)
        
        # Data tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚌 Top High-Risk Routes")
            top_routes = route_stats.head(10)
            st.dataframe(top_routes[['route_name', 'avg_delay_risk', 'trip_count', 'high_risk_pct']])
        
        with col2:
            st.subheader("🛑 Top High-Risk Stops")
            top_stops = stop_stats.head(10)
            st.dataframe(top_stops[['stop_name', 'avg_delay_risk', 'trip_count', 'high_risk_pct']]) 