"""
Configuration settings for the Course Notes Chatbot
"""

from pathlib import Path
import os

class Config:
    """Configuration class for chatbot settings"""
    
    def __init__(self):
        # Base directory (parent of code directory)
        self.BASE_DIR = Path(__file__).parent.parent
        
        # Data directories
        self.COURSE_NOTES_DIR = self.BASE_DIR / "course_notes"
        self.VECTOR_STORE_DIR = self.BASE_DIR / "vector_store"
        
        # Model settings
        self.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        self.LLM_MODEL = "distilbert-base-cased-distilled-squad"
        
        # Document processing settings
        self.CHUNK_SIZE = 500
        self.CHUNK_OVERLAP = 50
        
        # Query settings
        self.TOP_K_RESULTS = 5  # Increased for better context
        
        # Supported file extensions
        self.SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.md', '.xlsx'}
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        # Create directories
        self.COURSE_NOTES_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Only print if running interactively (not during imports)
        import sys
        if hasattr(sys, 'ps1') or 'jupyter' in sys.modules:
            print(f"✅ Directories ready:")
            print(f"   📁 Course notes: {self.COURSE_NOTES_DIR}")
            print(f"   📁 Vector store: {self.VECTOR_STORE_DIR}")