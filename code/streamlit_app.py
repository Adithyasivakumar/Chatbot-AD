"""
Streamlit Web Interface for Course Notes Chatbot
Simple web UI for interacting with the chatbot
"""

import streamlit as st
import sys
from pathlib import Path

# Add the code directory to the path
sys.path.append(str(Path(__file__).parent))

try:
    from chatbot import CourseNotesChatbot
    from config import Config
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please install all required dependencies using: pip install -r requirements.txt")
    st.stop()

def initialize_chatbot():
    """Initialize chatbot with caching"""
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing chatbot... This may take a few minutes."):
            chatbot = CourseNotesChatbot()
            if chatbot.initialize_chatbot():
                st.session_state.chatbot = chatbot
                st.session_state.initialized = True
                return True
            else:
                st.error("Failed to initialize chatbot")
                return False
    return st.session_state.get('initialized', False)

def main():
    st.set_page_config(
        page_title="Course Notes Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Course Notes Chatbot")
    st.markdown("Ask questions about your course notes!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("📚 About")
        st.markdown("""
        This chatbot answers questions based on your course notes using:
        - **LlamaIndex** for document processing
        - **Faiss** for vector search
        - **Hugging Face** for embeddings and LLM
        """)
        
        st.header("📄 Supported Formats")
        st.markdown("""
        - Text files (.txt)
        - PDF files (.pdf)
        - Word documents (.docx)
        - Markdown files (.md)
        - Excel files (.xlsx)
        """)
        
        config = Config()
        course_notes_files = list(config.COURSE_NOTES_DIR.glob("*"))
        
        st.header("📁 Your Documents")
        if course_notes_files:
            for file in course_notes_files:
                if file.suffix in config.SUPPORTED_EXTENSIONS:
                    st.markdown(f"✅ {file.name}")
        else:
            st.warning(f"No documents found in {config.COURSE_NOTES_DIR}")
            st.markdown("Please add your course notes to the `course_notes` folder.")
    
    # Check if documents exist
    config = Config()
    course_notes_files = list(config.COURSE_NOTES_DIR.glob("*"))
    supported_files = [f for f in course_notes_files if f.suffix in config.SUPPORTED_EXTENSIONS]
    
    if not supported_files:
        st.error("⚠️ No course notes found!")
        st.info(f"Please add your course notes to: `{config.COURSE_NOTES_DIR}`")
        st.info("Supported formats: " + ", ".join(config.SUPPORTED_EXTENSIONS))
        return
    
    # Initialize chatbot
    if not initialize_chatbot():
        return
    
    chatbot = st.session_state.chatbot
    
    # Display document information
    with st.expander("📊 Document Information"):
        doc_info = chatbot.get_document_info()
        st.json(doc_info)
    
    # Chat interface
    st.header("💬 Ask Questions")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your course notes..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot.ask_question(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sample questions
    st.header("💡 Sample Questions")
    sample_questions = [
        "What are the main topics covered in the course?",
        "Can you summarize the key concepts?",
        "What are the important definitions?",
        "Explain the main theories discussed",
        "What are the practical applications mentioned?"
    ]
    
    col1, col2 = st.columns(2)
    for i, question in enumerate(sample_questions):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(question, key=f"sample_{i}"):
                # Simulate clicking on the question
                st.session_state.messages.append({"role": "user", "content": question})
                with st.spinner("Thinking..."):
                    response = chatbot.ask_question(question)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.experimental_rerun()

if __name__ == "__main__":
    main()