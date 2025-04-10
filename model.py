from src.document import DocumentProcessor
from src.embeddings import VectorStore
from src.substack import SubstackConfig, SubstackProcessor
from src.llm import TransformersWrapper
from langchain.schema import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from os import getenv
from huggingface_hub import login

class RagModel:

    def __init__(self):
        self._load_credentials()
        self._init_components()
        self._setup_chains()

    def _load_credentials(self):
        """Загрузка учетных данных"""
        load_dotenv()
        login(token=getenv("HUGGINGFACE_TOKEN"))

    def _init_components(self):
        """Инициализация основных компонентов"""
        self.llm = TransformersWrapper('google/flan-t5-xl')
        self.vector_store = self._init_vector_store()

    def _init_vector_store(self):
        """Инициализация векторного хранилища"""
        config = SubstackConfig(
            target_substack='https://gonzoml.substack.com/', 
            query='deepseek',
            limit=5)
        
        processor = SubstackProcessor(config)
        processor.wide_search()

        
        documents = [Document(page_content=doc) for doc in processor.get_documents()]
        dp = DocumentProcessor(chunk_size=600, chunk_overlap=60)

        documents = dp.split_documents(documents) 

        vs = VectorStore()
        
        return vs.create_index(documents)

    def _setup_chains(self):
        """Настройка цепочек обработки"""
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_store,
            llm=self.llm,
            prompt=self._get_query_prompt()
        )

        self.rag_template = ChatPromptTemplate.from_template("""
            You are an extremely qualified specialist in machine learning and artificial intelligence.
            Answer the Question briefly using information from the Context.
                
            Question: {question}
            Context: {context}
            Answer:"""
        )
        
        self.no_rag_template = ChatPromptTemplate.from_template("""
            You are an extremely qualified specialist in machine learning and artificial intelligence.
            Answer the Question briefly.
            
            Question: {question}
            Answer:"""
        )

        self.rag_chain = (
            {"context": RunnablePassthrough() | self._safe_retrieve, 
             "question": RunnablePassthrough()}
            | self.rag_template
            | self.llm
            | StrOutputParser()
        )

        self.no_rag_chain = (
            {"question": RunnablePassthrough()}
            | self.no_rag_template
            | self.llm
            | StrOutputParser()
        )
    
    def _get_query_prompt(self):
        """Промпт для генерации поисковых запросов"""
        return PromptTemplate(
            input_variables=["question"],
            template="""
            You are an assistant for generating search queries. Generate 4 variants of a question formulation 
            to search for relevant documents in a vector database.
            Original question: {question}"""
        )

    def _safe_retrieve(self, question: str) -> str:
        """Безопасный поиск с обработкой ошибок"""
        
        docs = self.retriever.get_relevant_documents(question, k=5)
        if docs:
            return '\n'.join([i.page_content for i in docs])
        return "No relevant information found"
        
    def ask_rag(self, question: str) -> str:
        """Метод для выполнения запросов с RAG"""
        return self.rag_chain.invoke(question)
    
    def ask_no_rag(self, question: str) -> str:
        """Метод для выполнения запросов без RAG"""
        return self.no_rag_chain.invoke(question)