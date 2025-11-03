#!/usr/bin/env python3
"""
Quick Start Script for Course Notes Chatbot
Run this script to quickly test your chatbot setup
"""

import sys
from pathlib import Path

# Add code directory to path
sys.path.append(str(Path(__file__).parent / "code"))

def main():
    print("🤖 Course Notes Chatbot - Quick Start")
    print("=" * 50)
    
    try:
        from chatbot import CourseNotesChatbot
        
        print("📚 Initializing chatbot...")
        chatbot = CourseNotesChatbot()
        
        if not chatbot.initialize_chatbot():
            print("❌ Failed to initialize chatbot")
            print("💡 Make sure you have:")
            print("   1. Installed requirements: pip install -r requirements.txt")
            print("   2. Added course notes to course_notes/ folder")
            return
        
        print("✅ Chatbot ready!")
        
        # Test questions
        test_questions = [
            "What is machine learning?",
            "What are the types of machine learning?",
            "What is overfitting?"
        ]
        
        print(f"\n🧪 Testing with {len(test_questions)} questions:")
        print("-" * 50)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            answer = chatbot.ask_question(question)
            print(f"🤖 Answer: {answer[:200]}...")
        
        print("\n" + "=" * 50)
        print("🎉 Quick test completed!")
        print("💡 For full functionality, use:")
        print("   - python code/chatbot.py (command line)")
        print("   - streamlit run code/streamlit_app.py (web interface)")
        print("   - demo_notebook.ipynb (Jupyter notebook)")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please install requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()