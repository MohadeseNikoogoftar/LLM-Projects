# Chat with PDF using hybrid RAG
A simple RAG system using Deepseek, LangChain, and Streamlit to chat with PDFs and answer complex questions about your local documents. This project improves the accuracy of the RAG by using a hybrid approach of retrieval leveraging both semantic and bm25 search.

# Pre-requisites
Install Ollama on your local machine from the [official website](https://ollama.com/). And then pull the Deepseek model:

```bash
ollama pull deepseek-r1:14b
```

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```
- streamlit
- langchain_core
- langchain_community
- langchain_ollama
- pdfplumber
- rank_bm25
- nltk

# Run
Run the Streamlit app:

```bash
streamlit run hybrid_pdf_rag.py
```
