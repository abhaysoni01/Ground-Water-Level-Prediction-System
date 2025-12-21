#!/bin/bash
# Groundwater Prediction System - Quick Start Script

echo "🌊 Groundwater Level Prediction System"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Run the application
echo "🚀 Starting the application..."
echo "Access at: http://localhost:8501"
streamlit run app.py --server.port 8501 --server.address 0.0.0.0