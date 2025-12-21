@echo off
REM Groundwater Prediction System - Quick Start Script for Windows

echo 🌊 Groundwater Level Prediction System
echo ======================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Run the application
echo 🚀 Starting the application...
echo Access at: http://localhost:8501
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

pause