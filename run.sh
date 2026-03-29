#!/bin/bash

# Deepfake Detection - Quick Run Script

echo "🛡️ Deepfake Detection App"
echo "=========================="
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

echo "Starting Streamlit app..."
echo ""
echo "The app will open in your browser automatically."
echo "Press Ctrl+C to stop the server."
echo ""

streamlit run app.py
