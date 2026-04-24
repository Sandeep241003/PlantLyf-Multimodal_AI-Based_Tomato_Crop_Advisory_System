# 🧠 RAG-Based AI Chatbot (Local LLM + Voice Support)

An intelligent **Retrieval-Augmented Generation (RAG) chatbot** built using **local LLMs (Ollama)**, **vector search**, and **voice interaction (Whisper)** — designed to provide accurate, context-aware answers from custom documents.

---

## 🚀 Features

- 🔍 **Retrieval-Augmented Generation (RAG)**
  - Context-aware answers using document retrieval
- 🧠 **Local LLM Support (Ollama)**
  - No API cost, runs fully offline
- 🎙️ **Voice Input (Speech-to-Text)**
  - Powered by OpenAI Whisper (local)
- 💬 **Interactive Chat UI**
  - Built with Streamlit
- 📄 **Custom Knowledge Base**
  - Upload and query your own documents
- ⚡ **Fast Semantic Search**
  - Using vector embeddings
- 🔒 **Privacy Friendly**
  - Everything runs locally

---

## 🏗️ Tech Stack

- **Frontend/UI:** Streamlit  
- **LLM:** Ollama (Mistral / LLaMA3)  
- **Embeddings:** Sentence Transformers / Ollama embeddings  
- **Vector DB:** FAISS / ChromaDB  
- **Speech-to-Text:** Whisper (local)  
- **Audio Processing:** PyDub + FFmpeg  
- **Backend:** Python  

---

## 📁 Project Structure


RAG-Chatbot/
│── chatbot2.py # Main Streamlit app
│── rag_pipeline.py # RAG logic (retrieval + generation)
│── embeddings.py # Embedding generation
│── vector_store/ # FAISS / Chroma DB storage
│── data/ # Input documents
│── utils/ # Helper functions
│── audio/ # Temporary audio files
│── requirements.txt
│── README.md


---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Install Ollama

Download and install Ollama:
https://ollama.com/

Pull a model:

ollama pull mistral
5️⃣ Install FFmpeg (Required for Audio)

Download from:
https://ffmpeg.org/download.html

Set path in code if needed:

AudioSegment.converter = "path_to_ffmpeg/bin/ffmpeg.exe"
▶️ Running the App
streamlit run chatbot2.py
🎤 Voice Input Feature
Click Record Audio
Speak your query
Audio is converted using Whisper
Text is passed to RAG pipeline
🔄 How It Works
User asks a question (text or voice)
Query is converted into embeddings
Relevant documents are retrieved from vector DB
Context + query sent to LLM (Ollama)
LLM generates final response
🧩 RAG Pipeline Flow
User Query
   ↓
Embedding Model
   ↓
Vector Search (FAISS)
   ↓
Relevant Context
   ↓
LLM (Ollama)
   ↓
Final Answer
