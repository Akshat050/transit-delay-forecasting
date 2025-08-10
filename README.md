# 🚌 DelaySenseAI - Transit Delay Prediction System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**DelaySenseAI** is a machine learning-powered transit delay forecasting system that converts GTFS static schedules into actionable predictions. It helps transit authorities proactively manage reliability, target hotspots, and improve rider experience.

## 🌟 Features

- **🚀 Pre-trip Delay Forecasts**: Predict delay probability and magnitude for any route/trip/stop combination
- **🗺️ Interactive Hotspot Maps**: Visualize delay risk across your transit network
- **📊 Route-Specific Analysis**: Deep dive into individual route performance
- **⏰ Time-Based Patterns**: Identify peak hour bottlenecks and temporal trends
- **🔄 Transfer Reliability**: Predict connection success probability between routes
- **🧪 What-If Planning**: Simulate schedule changes and see predicted impacts
- **📈 Model Performance**: Comprehensive evaluation with calibration curves and SHAP analysis

## 🎯 Use Cases

- **Transit Planners**: Optimize schedules and identify high-risk segments
- **Operations Teams**: Proactively manage service reliability
- **Riders**: Get delay risk information for trip planning
- **Analysts**: Understand network performance patterns

## 🚀 Quick Start

### Option 1: Try Online (Recommended)
Visit our live demo: [DelaySenseAI Streamlit Cloud](https://delaysenseai.streamlit.app)

### Option 2: Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Akshat050/transit-delay-forecasting.git
   cd transit-delay-forecasting
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your GTFS data**
   Place your GTFS files in the project directory:
   - `routes.txt`
   - `trips.txt` 
   - `stop_times.txt`
   - `stops.txt`
   - `calendar.txt`
   - `calendar_dates.txt`

4. **Train the model**
   ```bash
   python main.py
   ```

5. **Launch the dashboard**
   ```bash
   streamlit run dashboard.py
   ```

## 📁 Project Structure

```
transit-delay-forecasting/
├── main.py                 # Main orchestration script
├── data_processor.py      # GTFS data processing and feature engineering
├── delay_predictor.py     # ML model training and prediction
├── visualizer.py          # Interactive visualizations and maps
├── transfer_processor.py  # Transfer reliability analysis
├── calibration.py         # Probability calibration utilities
├── scenario_utils.py      # What-if scenario simulation
├── dashboard.py           # Streamlit web application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── outputs/              # Generated models and reports
```

## 🧠 Machine Learning Model

- **Algorithm**: XGBoost with probability calibration
- **Features**: 50+ engineered features including temporal, spatial, and schedule-based patterns
- **Target**: Binary classification (delay > 3 minutes)
- **Performance**: Optimized for imbalanced datasets with AUC-PR and calibration metrics

### Key Features
- **Temporal**: Hour-of-day, peak flags, day-of-week patterns
- **Spatial**: Inter-stop distance, stop density, terminal proximity
- **Schedule**: Planned dwell time, headway analysis, run time patterns
- **Network**: Route frequency, stop popularity, transfer connections

## 📊 Sample Outputs

The system generates comprehensive reports including:
- Interactive delay hotspot maps
- Route-specific performance analysis
- Transfer reliability assessments
- Model performance metrics
- Calibration curves and SHAP explanations

## 🔧 Development

### Prerequisites
- Python 3.11+
- GTFS data files
- Required Python packages (see requirements.txt)

### Installation
```bash
# Clone the repository
git clone https://github.com/Akshat050/transit-delay-forecasting.git
cd transit-delay-forecasting

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run dashboard.py
```

## 📈 Performance

The system provides:
- **Real-time predictions** with <100ms latency
- **High accuracy** with AUC-PR > 0.85
- **Scalable architecture** supporting large transit networks
- **Interactive visualizations** for data exploration

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- GTFS specification for transit data standards
- Streamlit for the web application framework
- XGBoost for the machine learning algorithm
- Open-source community for various supporting libraries
