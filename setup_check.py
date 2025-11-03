#!/usr/bin/env python3
"""
Setup checker for Course Notes Chatbot
Diagnoses common issues and provides solutions
"""

import sys
import subprocess
from pathlib import Path
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} - NOT INSTALLED")
        return False

def check_dependencies():
    """Check all required dependencies"""
    print("\n📦 Checking dependencies:")
    
    dependencies = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("sentence-transformers", "sentence_transformers"),
        ("llama-index", "llama_index"),
        ("faiss-cpu", "faiss"),
        ("PyPDF2", "PyPDF2"),
        ("python-docx", "docx"),
        ("pandas", "pandas"),
        ("openpyxl", "openpyxl"),
        ("numpy", "numpy")
    ]
    
    missing = []
    for package, import_name in dependencies:
        if not check_package(package, import_name):
            missing.append(package)
    
    return missing

def check_project_structure():
    """Check if project structure is correct"""
    print("\n📁 Checking project structure:")
    
    base_dir = Path(__file__).parent
    required_dirs = ["code", "course_notes", "vector_store"]
    required_files = ["code/config.py", "code/document_loader.py", "code/chatbot.py"]
    
    issues = []
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ - MISSING")
            issues.append(f"Create directory: {dir_name}")
    
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            issues.append(f"Create file: {file_path}")
    
    return issues

def check_course_notes():
    """Check if course notes are available"""
    print("\n📚 Checking course notes:")
    
    course_notes_dir = Path(__file__).parent / "course_notes"
    
    if not course_notes_dir.exists():
        print("❌ course_notes/ directory not found")
        return ["Create course_notes directory and add your notes"]
    
    files = list(course_notes_dir.glob("*"))
    if not files:
        print("⚠️  No files found in course_notes/")
        return ["Add your course notes to course_notes/ directory"]
    
    supported_extensions = {'.txt', '.pdf', '.docx', '.md', '.xlsx'}
    supported_files = [f for f in files if f.suffix.lower() in supported_extensions]
    
    print(f"📄 Found {len(files)} total files")
    print(f"✅ Found {len(supported_files)} supported files")
    
    if supported_files:
        for file in supported_files[:5]:  # Show first 5
            print(f"   • {file.name}")
        if len(supported_files) > 5:
            print(f"   ... and {len(supported_files) - 5} more")
    
    if not supported_files:
        return ["Add supported files (.txt, .pdf, .docx, .md, .xlsx) to course_notes/"]
    
    return []

def provide_solutions(missing_packages, structure_issues, notes_issues):
    """Provide solutions for found issues"""
    if not (missing_packages or structure_issues or notes_issues):
        print("\n🎉 Everything looks good! You can run the chatbot now.")
        print("\n🚀 To start the chatbot:")
        print("   python run_chatbot.py")
        return
    
    print("\n🔧 Solutions:")
    
    if missing_packages:
        print("\n1. Install missing packages:")
        print("   pip install " + " ".join(missing_packages))
        print("\n   Or install from requirements.txt:")
        print("   pip install -r requirements.txt")
    
    if structure_issues:
        print("\n2. Fix project structure:")
        for issue in structure_issues:
            print(f"   • {issue}")
    
    if notes_issues:
        print("\n3. Add course notes:")
        for issue in notes_issues:
            print(f"   • {issue}")
        print("\n   Supported formats:")
        print("   • Text files: .txt, .md")
        print("   • PDF files: .pdf")
        print("   • Word documents: .docx")
        print("   • Excel files: .xlsx")

def main():
    print("🔍 Course Notes Chatbot - Setup Checker")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check dependencies
    missing_packages = check_dependencies()
    
    # Check project structure
    structure_issues = check_project_structure()
    
    # Check course notes
    notes_issues = check_course_notes()
    
    # Provide solutions
    provide_solutions(missing_packages, structure_issues, notes_issues)

if __name__ == "__main__":
    main()
