from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOllama
from langchain.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.utils import filter_complex_metadata


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        """
        Initializes the question-answering system with default configuration.

        The constructor sets up the following components:
        - model: ChatOllama LLM model ('neural_chat')
        - text_splitter: RecursiveCharacterTextSplitter for splitting text into chunks with overlap
        - prompt_template: PromptTemplate for building prompt with input variables for question and context.
        """
        # Initialize Ollama model with 'neural-chat'
        self.model = ChatOllama(model='neural-chat')
        # Initialize RecursiveCharacterTextSplitter with chunk_size and overlap
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # Initialize PromptTemplate with a predefined template and placeholders for question and context
        self.prompt = PromptTemplate.from_template(
            """
            You are a helpful assistant that analyses PDF files content to give answers to provided questions.
            Use the following pieces of context retrieved from the files to answer the question.
            If you can't find the answer to the question, or you just don't know the answer, just say that you don't know. 
            Don't hallucinate.

            Question: {question}
            Context: {context}
            Answer:
            """
        )

    
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
                "k": 3,
                "score_threshold": 0.5
            }
        )

        # Define a processing chain for handling question-answers
        # Chain is defined as:
        # 1. get "context" from the retriever
        # 2. a passthrough for the "question"
        # 3. Fill in the prompt
        # 4. Invoke the LLM model
        # 5. Parse output
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )


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
        
        return self.chain.invoke(query)
    
    
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