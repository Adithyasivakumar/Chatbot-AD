# Course Notes Chatbot

A RAG-based chatbot that answers questions from your course notes using LlamaIndex, Faiss, and Hugging Face transformers.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 2. Add Your Course Notes

Put your course notes in the `course_notes/` directory:
- Supported formats: `.txt`, `.pdf`, `.docx`, `.md`, `.xlsx`
- The chatbot will automatically process all files in this directory

```bash
# Example: Add your notes
cp /path/to/your/notes.pdf course_notes/
cp /path/to/your/lecture.txt course_notes/
```

### 3. Run the Chatbot

**Option A: Using the launcher script (Recommended)**
```bash
python run_chatbot.py
```

**Option B: Direct execution**
```bash
cd code
python chatbot.py
```

**Option C: Using the demo notebook**
```bash
jupyter notebook demo.ipynb
```

## 📁 Project Structure

```
Chatbot AD/
├── code/
│   ├── chatbot.py          # Main chatbot implementation
│   ├── config.py           # Configuration settings
│   └── document_loader.py  # Document loading utilities
├── course_notes/           # Put your course notes here
├── vector_store/           # Vector database storage
├── requirements.txt        # Python dependencies
├── run_chatbot.py         # Simple launcher script
├── demo.ipynb             # Jupyter demo notebook
└── README.md              # This file
```

## 🤖 Usage

1. **Start the chatbot**: Run `python run_chatbot.py`
2. **Ask questions**: Type your questions about the course material
3. **Get answers**: The chatbot will search through your notes and provide relevant answers
4. **Exit**: Type `quit`, `exit`, or press Ctrl+C to stop

## Example Session

```
🤖 Course Notes Chatbot
==================================================
📚 Initializing chatbot with your course notes...
✅ Chatbot ready!

Document Info:
📄 Loaded 25 document chunks

==================================================
💬 You can now ask questions about your course notes!
Type 'quit' or 'exit' to stop
==================================================

🤔 Your question: What is machine learning?

🤖 Thinking...

💡 Answer:
Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed...

🤔 Your question: quit
👋 Goodbye!
```

## 🔧 Configuration

Edit `code/config.py` to customize:
- Chunk size for document processing
- Number of search results
- Model selection
- Directory paths

## 🐛 Troubleshooting

**No documents found?**
- Make sure your course notes are in the `course_notes/` directory
- Check that file formats are supported

**Installation issues?**
- Try: `pip install --upgrade pip`
- For M1 Mac users: `pip install faiss-cpu` instead of `faiss-gpu`

**Memory errors?**
- Reduce chunk size in `config.py`
- Use smaller document files

## 📝 Supported File Formats

- **Text files**: `.txt`, `.md`
- **PDF files**: `.pdf`
- **Word documents**: `.docx`
- **Excel files**: `.xlsx`