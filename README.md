# Groundwater Level Prediction System - GitHub Repository

## 🌊 Advanced ML Platform for Punjab District Water Resource Forecasting

A comprehensive machine learning application for predicting groundwater levels across Punjab districts using LSTM neural networks. Built with Streamlit, TensorFlow, and modern data visualization techniques.

### 🚀 Features

- **LSTM Neural Networks**: Advanced time series forecasting for accurate groundwater predictions
- **Multi-District Support**: 22 Punjab districts with individual trained models
- **Interactive Dashboard**: Real-time system overview and district analytics
- **Advanced Analytics**: Confidence intervals, trend analysis, and performance metrics
- **Professional UI**: Academic-grade interface with dark/light theme support
- **Data Export**: Download predictions and analysis results
- **Real-time Predictions**: 3-6 month forecasting horizons

### 📊 Technical Specifications

- **Framework**: Streamlit Web Application
- **ML Model**: LSTM Neural Networks (TensorFlow/Keras)
- **Data**: 2018-2024 Punjab groundwater measurements
- **Features**: Historical levels, rainfall, soil moisture
- **Performance**: RMSE, MAE, R² evaluation metrics
- **Visualization**: Plotly interactive charts

### 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/groundwater-prediction-system.git
   cd groundwater-prediction-system
   ```

2. **Create virtual environment** (Note: Use Python 3.11 or earlier, as TensorFlow does not support Python 3.12):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Access the app** at `http://localhost:8501`

### 📁 Project Structure

```
groundwater-prediction-system/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── data/
│   └── processed_data.csv # Processed groundwater data
├── models/               # Trained LSTM models
│   ├── Amritsar_model.h5
│   ├── Amritsar_scaler.pkl
│   └── ... (22 district models)
└── src/
    └── prediction.py     # Forecasting functions
```

### 🎯 Usage

1. **Dashboard**: View system overview and district statistics
2. **Analysis**: Select district and forecast horizon, generate predictions
3. **Predictions**: View forecast results with confidence intervals
4. **Methodology**: Learn about the ML approach and technical details
5. **Data Export**: Download predictions and performance metrics

### 📈 Model Performance

- **RMSE**: < 0.5 meters (district-dependent)
- **MAE**: < 0.3 meters average error
- **R² Score**: > 0.85 for most districts
- **Training Data**: 2018-2023 historical records
- **Test Data**: 2024 validation period

### 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments

- Punjab Government Water Resources Department
- Academic research in Environmental Machine Learning
- TensorFlow/Keras and Streamlit communities

### 📞 Contact

For questions or collaborations:
- **Project**: Groundwater Level Prediction System
- **Institution**: Academic Research Project
- **Focus**: Environmental Data Science & Water Resource Management

---

**🏛️ Academic Project** | Department of Computer Science & Engineering | Advanced Machine Learning for Environmental Resource Management