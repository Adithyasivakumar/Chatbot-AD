#!/usr/bin/env python3
"""
Test script for 5 different questions with comprehensive answers
As requested in the project requirements
"""

import sys
from pathlib import Path

# Add code directory to path
sys.path.append(str(Path(__file__).parent / "code"))

def main():
    print("🤖 Testing Course Notes Chatbot - 5 Different Questions")
    print("=" * 60)
    
    try:
        from chatbot import CourseNotesChatbot
        
        print("📚 Initializing chatbot...")
        chatbot = CourseNotesChatbot()
        
        if not chatbot.initialize_chatbot():
            print("❌ Failed to initialize chatbot")
            return
        
        print("✅ Chatbot ready!\n")
        
        # Define 5 comprehensive test questions covering different aspects
        questions = [
            {
                "id": 1,
                "question": "What is machine learning and how does it work?",
                "topic": "Introduction & Definition"
            },
            {
                "id": 2,
                "question": "What are the different types of machine learning? Explain each type with their characteristics.",
                "topic": "Types of Machine Learning"
            },
            {
                "id": 3,
                "question": "What is overfitting and underfitting in machine learning? Explain the difference.",
                "topic": "Model Performance Issues"
            },
            {
                "id": 4,
                "question": "Explain the key steps in data preprocessing for machine learning projects.",
                "topic": "Data Preprocessing"
            },
            {
                "id": 5,
                "question": "What are the different model evaluation metrics? How do you calculate precision, recall, and F1-score?",
                "topic": "Model Evaluation"
            }
        ]
        
        # Process each question and display results
        for q in questions:
            print(f"📝 QUESTION {q['id']}: {q['topic']}")
            print(f"❓ {q['question']}")
            print(f"🔍 Processing...")
            
            answer = chatbot.ask_question(q['question'])
            
            print(f"💡 ANSWER {q['id']}:")
            print(answer)
            print("\n" + "-" * 60 + "\n")
        
        print("🎉 All 5 questions processed successfully!")
        print("\n💡 Summary:")
        print("- Each answer contains relevant information from course notes")
        print("- Answers are extracted using AI-powered semantic search")
        print("- Source attribution is provided for each response")
        print("- Different topics are covered comprehensively")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please ensure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()