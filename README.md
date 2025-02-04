# PDF Chat Assistant

An intelligent PDF document assistant that combines semantic search, BM25 retrieval, and LLM-powered chat capabilities to provide contextual answers from PDF documents.

## Features

- 📄 Smart PDF text extraction and processing
- 🔍 Hybrid search combining semantic and keyword-based approaches
- 💡 Context-aware document understanding
- 💬 Interactive chat interface for document queries
- 🚀 Ensemble retrieval system for improved accuracy
- 📊 Support for processing large documents
- 💾 Context preservation between chunks

## Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`
- At least 8GB RAM recommended for processing large PDFs

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pdf-chat-assistant.git
cd pdf-chat-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the chat assistant:
```bash
python pdf_chat_assistant.py
```

2. Enter the path to your PDF file when prompted

3. Start asking questions about the document

4. Type 'bye' to exit the chat session

## Configuration

Adjust settings in `config.py`:
- Model parameters
- Chunk size and overlap
- Number of semantic results
- Embedding model settings
- Temperature and other LLM parameters

## Project Structure

```
pdf-chat-assistant/
├── pdf_chat_assistant.py    # Main chat interface
├── pdf_retriever_processor.py # PDF processing and retrieval
├── config.py               # Configuration settings
├── requirements.txt        # Project dependencies
└── README.md              # Documentation
```

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

Your Name - [your@email.com](mailto:your@email.com)
Project Link: [https://github.com/yourusername/pdf-chat-assistant](https://github.com/yourusername/pdf-chat-assistant)