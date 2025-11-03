#!/usr/bin/env python3
"""
Simple script to run the Course Notes Chatbot
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'llama_index', 'faiss', 'torch', 'transformers', 
        'sentence_transformers', 'PyPDF2', 'docx', 'openpyxl'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    return missing

def main():
    print("🚀 Course Notes Chatbot Launcher")
    print("=" * 40)
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n💡 To install dependencies, run:")
        print("   pip install -r requirements.txt")
        return 1
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir / "code"))
    
    try:
        from chatbot import main as chatbot_main
        chatbot_main()
    except ImportError as e:
        print(f"❌ Error importing chatbot: {e}")
        print("💡 Make sure all files are in the correct location")
        return 1
    except Exception as e:
        print(f"❌ Error running chatbot: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
