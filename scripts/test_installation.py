#!/usr/bin/env python3
"""
Test script to verify the installation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas")
        
        import numpy as np
        print("✅ numpy")
        
        import sqlalchemy
        print("✅ sqlalchemy")
        
        import networkx as nx
        print("✅ networkx")
        
        import torch
        print("✅ torch")
        
        import torch_geometric
        print("✅ torch_geometric")
        
        import sklearn
        print("✅ scikit-learn")
        
        import openai
        print("✅ openai")
        
        import streamlit
        print("✅ streamlit")
        
        import plotly
        print("✅ plotly")
        
        print("\n✅ All packages imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from src.config import Config
        print("✅ Configuration loaded")
        
        if Config.OPENAI_API_KEY and Config.OPENAI_API_KEY != "your_openai_api_key_here":
            print("✅ OpenAI API key configured")
        else:
            print("⚠️  OpenAI API key not configured")
        
        print(f"✅ Database URL: {Config.DATABASE_URL}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_csv_file():
    """Test that the CSV file exists."""
    print("\nTesting CSV file...")
    
    if os.path.exists("scotiabank.csv"):
        print("✅ scotiabank.csv found")
        
        # Quick peek at the data
        try:
            import pandas as pd
            df = pd.read_csv("scotiabank.csv", header=None, nrows=5)
            print(f"✅ CSV file readable, {len(df)} sample rows loaded")
            return True
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return False
    else:
        print("❌ scotiabank.csv not found")
        return False

def main():
    print("🧪 Running installation tests...\n")
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_config():
        tests_passed += 1
    
    if test_csv_file():
        tests_passed += 1
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("1. Configure your OpenAI API key in .env file")
        print("2. Start PostgreSQL service")
        print("3. Run: python scripts/init_db.py")
        print("4. Run: python scripts/load_data.py")
        print("5. Run: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
