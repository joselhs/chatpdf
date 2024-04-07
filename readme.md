# ChatPDF
Simple RAG app to upload PDF files to a local vector store and ask questions using LLMs (Ollama neural-chat or OpenAI gpt 3.5). The app has been developed using:
- Langchain: to manage Prompts and LLMs interactions and retrieval.
- Chroma: ChromaDB as vector store to retrieve information from PDF files.
- Ollama: to use local LLM models.
- Streamlit: to create App interface. 


## Set up
To make it run with OpenAI models, an OpenAI API Key is needed.

Create a .env file and store the API Key as:

OPENAI_API_KEY='sk-xxxxxx'


## Run
To Run the Streamlit App with OpenAI model, run the following command in terminal:

*streamlit run app.py -- openai*


To Run it using Ollama neural-chat, Ollama must be installed and neural-chat model downloaded. Run it with the following commands:

*streamlit run app.py*

or 

*streamlit run app.py -- neural-chat*