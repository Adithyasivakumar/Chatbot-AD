#!/bin/bash

echo "🚀 Installing Course Notes Chatbot Dependencies"
echo "================================================"

# Update pip first
echo "📦 Updating pip..."
python3 -m pip install --upgrade pip

# Install dependencies one by one for better error handling
echo "🔧 Installing core dependencies..."

echo "Installing torch..."
python3 -m pip install torch>=1.13.0

echo "Installing transformers..."
python3 -m pip install transformers>=4.25.0

echo "Installing sentence-transformers..."
python3 -m pip install sentence-transformers>=2.2.0

echo "Installing llama-index..."
python3 -m pip install llama-index>=0.9.0

echo "Installing faiss-cpu..."
python3 -m pip install faiss-cpu>=1.7.4

echo "Installing document processing packages..."
python3 -m pip install PyPDF2>=3.0.0
python3 -m pip install python-docx>=0.8.11
python3 -m pip install pandas>=1.5.0
python3 -m pip install openpyxl>=3.0.0

echo "Installing utilities..."
python3 -m pip install numpy>=1.21.0
python3 -m pip install markdown>=3.4.0

echo "✅ All dependencies installed!"
echo "🔍 Run 'python3 setup_check.py' to verify installation"
