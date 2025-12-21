#!/usr/bin/env python3
"""
Groundwater Prediction System - Quick Test Script
Tests basic functionality of the application components
"""

import sys
import os
import pandas as pd

def test_data_loading():
    """Test if data can be loaded"""
    try:
        df = pd.read_csv('data/processed_data.csv')
        print(f"✅ Data loaded: {df.shape[0]} records, {len(df['district'].unique())} districts")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_models():
    """Test if models can be discovered"""
    try:
        model_files = [f.replace('_model.h5', '') for f in os.listdir('models') if f.endswith('_model.h5')]
        print(f"✅ Found {len(model_files)} trained models")
        return len(model_files) > 0
    except Exception as e:
        print(f"❌ Model discovery failed: {e}")
        return False

def test_imports():
    """Test if required packages can be imported"""
    try:
        import streamlit as st
        import tensorflow as tf
        import plotly
        print("✅ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    print("🧪 Testing Groundwater Prediction System")
    print("=" * 50)

    tests = [
        ("Data Loading", test_data_loading),
        ("Model Discovery", test_models),
        ("Package Imports", test_imports)
    ]

    passed = 0
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1

    print(f"\n📊 Test Results: {passed}/{len(tests)} passed")

    if passed == len(tests):
        print("🎉 All tests passed! System is ready for deployment.")
        print("\n🚀 To run the application:")
        print("   python -m streamlit run app.py")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())