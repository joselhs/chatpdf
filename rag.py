import utils
import sys
from dotenv import load_dotenv

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        """
        Initializes the question-answering system with default configuration.

        The constructor sets up the following components:
        - model: ChatOllama LLM model ('neural_chat') or ChatOpenAI model
        - text_splitter: RecursiveCharacterTextSplitter for splitting text into chunks with overlap
        """
        # Check arguments for getting model. If not specified, use neural-chat
        if len(sys.argv) > 1:
            model = sys.argv[1]
        else:
            model = "neural-chat"

        if model == 'openai':
            # Initialize ChatOpenAI model
            # Load env vars
            load_dotenv()
            self.model = ChatOpenAI()
        else:
            # Initialize Ollama model with 'neural-chat'
            self.model = ChatOllama(model=model)
        # Initialize RecursiveCharacterTextSplitter with chunk_size and overlap
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    

    def ingest(self, pdf_file_path: str) -> None:
        """
        Ingests data from a PDF file, process the data and set up the different components.

        Parameters:
        - pdf_file_path (str): Path to the PDF file

        Usage:
        obj.ingest("path/to/data.pdf")

        This function uses a PyPDFLoader to load the data from the specific PDF file.

        Args:
        - file.path (str): Path to the PDF file
        """
        # Create a PDF loader
        loader = PyPDFLoader(file_path=pdf_file_path)

        # Load PDF file and split document in chunks
        chunks = loader.load_and_split(text_splitter=self.text_splitter)

        # Creates a ChromaDB vector store using embeddings
        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())

        # set up the retriever
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.5
            }
        )

        # Create RetrievalQA with Sources Chain by using previous instantiated model and retriever
        retrieval_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.model,
            retriever=self.retriever,
            return_source_documents=True,
            verbose=True
        )
        
        # set up the chain
        self.chain = retrieval_chain



    def ask(self, query: str) -> str:
        """
        Asks a question using the configured chain.

        Parameters:
        - query (str): Question to be asked

        Returns:
        - str: The result of processing the question through the chain.
        If the processing chain is not set up (empty), a message is returned asking to add a PDF file first.
        """
        if not self.chain:
            return "Please, add a PDF file first."
        else:
            response = self.chain.invoke(query)
            # For testing purpose
            for document in response['source_documents']:
                print("#########################")
                print(document.metadata)
                print(document.page_content)

            formated_response = utils.format_response(response)
        
        return formated_response
    
    
    def clear(self) -> None:
        """
        Clear the components in the question-answering system.

        This method resets the vector store, retriever and chain to None.
        effectively clearing the existing configuration.
        """
        # Set the vector store to None
        self.vector_store = None

        # Set the text splitter to None
        self.text_splitter = None

        # Set the chain to None
        self.chain = None