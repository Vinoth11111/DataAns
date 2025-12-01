from langchain_community.document_loaders import PyMuPDFLoader,DirectoryLoader,WebBaseLoader,PyPDFDirectoryLoader,CSVLoader,JSONLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import bs4
import time
import torch
import numpy as np
from tqdm import tqdm
import chromadb
import logging
import os
# setting up logging configuration
logging.basicConfig(level=logging.INFO, # setting log level to INFO
                    format='%(asctime)s - %(levelname)s - %(message)s', # format of the log message
                    handlers=[logging.FileHandler('logs_file.log'),# create log file
                              logging.StreamHandler()])# send message to the console

# initializing logger
logger = logging.getLogger(__name__)

# setting up chromadb instance
model = 'all-MiniLm-L6-v2'

embedding_model = HuggingFaceEmbeddings(model_name = model,model_kwargs = {'device': 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'},
    encode_kwargs = {'normalize_embeddings': False})
# creating the server instance for docker deployments
# initialize the host server 
host_db = os.environ.get('chromadb_server','localhost')
retries_count = 3
while retries_count > 0:
    try:
        client_i = chromadb.HttpClient(host=host_db,port=8000)# 8000 is default port for chromadb server.
        ChromaDB = Chroma(client=client_i,embedding_function= embedding_model)
        time.sleep(5)  # wait for 5 seconds before proceeding
        logger.info(f"Successfully connected to ChromaDB server at {host_db}:8000)")
        break
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB server at {host_db}:8000. Retrying...{retries_count} times", exc_info=True)
        retries_count -= 1
        if retries_count == 0:
            raise e

class ingestion:
    def __init__(self,model_name = model,embedding_model = embedding_model,VectorDB = ChromaDB):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.vectorDB = VectorDB

    # data loading methods
    def data_load(self,data_path):
        if data_path:
            print('Loading Book or Research Data...')
            logger.info(f'Loading Book Data from path: {data_path}')
            books = DirectoryLoader(data_path,glob='**/*.pdf',loader_cls=PyMuPDFLoader)
            books_data = books.load()
            logger.info('Book Data loaded successfully.')
            print(f'Number of documents loaded: {len(books_data)}')
            return books_data
        else:
            logger.error(f'Invalid data path specified: {data_path}',exc_info=True)# exc_info for seeing whole error.
            raise Exception('Invalid data path specified.')
        
        # this load() return data like this documents = [Document(page_content= 'text',metadata = {})]
    def research_data_load(self,data_path,type):
        if type == 'research':
            print('Loading research Data...')
            logger.info(f'Loading Interview Data from path: {data_path}')
            interviews = PyPDFDirectoryLoader(data_path)
            interviews_data = interviews.load()
            print(f'Number of research documents loaded: {len(interviews_data)}')
            logger.info('Research Data loaded successfully.')
            return interviews_data
        else:
            logger.error(f'Invalid data type specified: {type}')
            raise Exception('Invalid data type specified.')
    
    def article_data_load(self,web_path,type): # type for secondary validation.
        if type == 'article':
            print('Loading Article Data...')
            logger.info(f'Loading Article Data from URL: {web_path}')
            strainer = bs4.SoupStrainer('div',attrs={'class': 'text'})
            artical_data = WebBaseLoader(web_path,bs_kwargs={'parse_only': strainer})
            logger.info('Article Data loaded successfully.')
            return artical_data.load()
        else:
            logger.error(f'Invalid data type specified: {type}',exc_info=True)
            raise Exception('Invalid data type specified.')
    
    # other data loading methods like csv, json,txt can be added here.
    def other_data_load(self,data_path,type,json_line=None):
        print('Loading Other Data...')
        logger.info(f'Loading Other Data from path: {data_path}')
        if type == 'csv':
            other_data = CSVLoader(data_path,csv_args={'delimiter': ',', 'quotechar': '"'},autodetect_encoding=True)
            logger.info('Csv Data loaded successfully.')
            return other_data.load()
        elif type == 'json':
            other_data = JSONLoader(data_path,json_lines=json_line,json_path = '$.data[*]',autodetect_encoding=True)
            logger.info('Json Data loaded successfully.')
            return other_data.load()
        elif type == 'txt':
            other_data =TextLoader(data_path,autodetect_encoding=True)
            logger.info('Txt Data loaded successfully.')
            return other_data.load()
        #we can add more data types like txt, docx etc.
        
        
    # adding new data methods
    def add_book(book_path):
        try:
            logger.info(f'Adding new book from path: {book_path}')
            print('Adding new book to the dataset...')
            logger.info(f'Adding new book from path: {book_path}')
            new_book = PyMuPDFLoader(book_path)
            new_book_data = new_book.load()
            return new_book_data
        except Exception as e:
            logger.error(f'Error adding new book: {e}',exc_info=True)
            raise Exception('Invalid data type specified.')

    def add_interview(interview_path):
        try:
            logger.info(f'Adding new interview from path: {interview_path}')
            print('Adding new interview to the dataset....')
            new_interview = PyMuPDFLoader(interview_path)
            new_interview_data = new_interview.load()
            return new_interview_data
        except Exception as e:
            logger.error(f'Error adding new interview: {e}',exc_info=True)
            raise Exception('Invalid data type specified.')
    
    # data chunking methods
    # since load() return list of documents we need to use split_documents method, rather then split_text method.
    def book_research_chunking(self,books_data):
        try:
            logger.info('Chunking Book Data...')
            print('Chunking Book Data....')
            text_splitter = RecursiveCharacterTextSplitter(chunk_size= 500,chunk_overlap=100)
            # since we are using document loaders insead of raw text like page_content = 'text',metedata = {}, we need to use the split_text method
            """for i,content in enumerate(text_splitter.split_documents(books_data)[:5]):
                    print('chunk ',i+1)
                    print('-'*20)
                    print(content.page_content)"""
            return text_splitter.split_documents(tqdm(books_data))
        except Exception as e:
            logger.error(f'Error chunking book data: {e}',exc_info=True)
            raise Exception('Error chunking book data.')
        

    def interview_web_chunking(self,datas):
        try:
            print('Chunking Interview/Web Data....')
            print('No of documents to be chunked:',len(datas))
            logger.info(f'Chunking Interview/Web Data.... with {len(datas)} documents', )
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 250, chunk_overlap = 50)
            """for i,content in enumerate(text_splitter.split_documents(datas)[:5]):
                    print('chunk ',i+1)
                    print('-'*20)
                    print(content.page_content)"""
            return text_splitter.split_documents(tqdm(datas))
        
        except Exception as e:
            logger.error(f'Error chunking interview/web data: {e}',exc_info=True)
            raise Exception('Error chunking interview/web data.')
    
    # embedding methods
    # we dont have to emded chunks, because as for rag we want both content + metadata + index.
    # if we manually embed chunks then we will loose the metadata and index information.
    # note(for me): i created embedding_chunks() and exectuted got replace('/n) error and chnaged my approches to directly adding chunked data to chromadb.
    # this embedding chunks method is kept for reference purpose only.
    def embedding_chunks(self,chunked_book_data):
        try:
            print('Embedding book Data')
            print('No of chunks to be embedded:',len(chunked_book_data))
            logger.info(f'Embedding book Data with {len(chunked_book_data)} chunks')
            # we have set normalize_embeddings to False because sentenceformer already produce scaled vectors.
            embedding_model = self.embedding_model
            return embedding_model.embed_documents(tqdm(chunked_book_data.page_content))
        
        except Exception as e:
            logger.error(f'Error embedding book data: {e}',exc_info=True)
            raise Exception(f'Error embedding book data. {e}')
        
    # adding data to chromadb
    # in chromadb we can directly add the chucks by using documnets parameter in add_documents method.
    # it preserve both metadata and index information.
    def adding_data_to_chromadb(self,chunked_data):
        try:
            print('creating chromadb instance....')
            print('Persisting data to disk....')
            logging.info('Adding Data to the ChromaDB instance...')
            self.vectorDB.add_documents(documents=chunked_data)
            print('Data persisted successfully.')
        except Exception as e:
            print(f'Error adding data to ChromaDB: {e}')
            logging.error(f'Error adding data to ChromaDB: {e}',exc_info=True)
    
    # check data quality
    # checking for max and min length of the chunks,avg chunk size, blank pages with page, 
    def data_quality_report(self,data_path,type):
        try:
            logger.info('Generating data quality report...')
            blank_doc_count = 0
            blank_doc_dir = []
            if type == 'article':
                documents = self.article_data_load(data_path,type)
            elif type == 'research':
                documents = self.research_data_load(data_path,type)
            elif type == 'csv' or type == 'json':
                documents = self.other_data_load(data_path,type)
            else:
                documents = self.data_load(data_path)
            for page in documents:
                if not page.page_content or page.page_content.isspace():
                    blank_doc_count += 1
                    blank_doc_dir.append(page.metadata.get('source', 'Unknown Source'))
            for blank_dir in blank_doc_dir[:5]:
                logger.warning(f'Blank document found at: {blank_dir}')
            logger.info(f'Total blank documents found: {blank_doc_count}')

            # further quality checks can be added here like max,min,avg chunk size etc.
            
            # chunking for articles and interviews
            if type == 'article' or type == 'interview':
                chunks = self.interview_web_chunking(documents)
            else:
                chunks = self.book_research_chunking(documents)

            if not chunks:
                logger.warning('No chunks were created from the documents.')
                return

            chunk_len = [len(i.page_content) for i in chunks]
            total_chunks = len(chunk_len)
            max_chunk_size = max(chunk_len)
            min_chunk_size = min(chunk_len)
            avg_chunk_size = np.mean(chunk_len)   
            median_chunk_size = np.median(chunk_len)
            logger.warning(f'Total chunks created: {total_chunks}')
            logger.warning(f'Maximum chunk size: {max_chunk_size}' )
            logger.warning(f'Minimum chunk size: {min_chunk_size}')
            logger.warning(f'Average chunk size: {avg_chunk_size:.2f}')
            logger.warning(f'Median chunk size: {median_chunk_size:.2f}')     

            # returning first 5 chunks.
            logger.info(' first 5 chunks-------------')
            for i,content in enumerate(chunks[:5]):
                logger.info(f'chunk {i+1}')
                logger.info(content.metadata.get('source', 'Unknown Source'))
                logger.info(content.page_content[:250]) # printing first 250 characters of each chunk
            logger.info('Data quality report generated successfully.')
        except Exception as e:
            logger.error(f'Error checking data quality: {e}',exc_info=True)
    

# we are going to use a single vector file so creating a collection is not needed
"""
chroma/
    chroma.sqlite3
    index/
    file1/(persistent)
    file2/(persistent)
    chroma_db/(persistent)

in this case we explicitly set the collection name to 'chroma_db' to use a single vector database file.
eg: use Chroma(Collection_name = 'chroma_db')
which need to set manually or through predicting, which configure that the category query belongs to via llm.

think persit_directory as a folder where all the vector db files(collection_name) are stored.
"""
#vectorDB_config = Chroma(Collection_name = 'chroma_db',presistent_directory = 'chroma/',embedding_function= embedding_model)
"""
# Testing 1
# reading book data
Booked_book = ingestion(data_path = 'documents/books',data_type='book').book_huge_load()

# loading research papaers 
research_data = ingestion(data_path='documents/research_paper',data_type='research_paper').book_huge_load()

# loading intervieW DATA
interview_data = ingestion(data_path = 'documents/interview',data_type='interview').interview_data_load()

# loading article data
article_data = ingestion(data_path = 'https://www.geeksforgeeks.org/nlp/advanced-natural-language-processing-interview-question/',data_type='article').article_data_load()

# chunking book data

#chunking article data
chunked_article_data = ingestion(None,None).interview_web_chunking(article_data)
print('No of articles loaded:',len(article_data))
print(f'Number of article chunks: {len(chunked_article_data)}')

# Testing 2
ingestor = ingestion()
print(ingestor.data_quality_report('https://www.geeksforgeeks.org/nlp/advanced-natural-language-processing-interview-question/','article'))
"""

#testing 3
# creating the ingestion object
ingestor = ingestion()
# loading the data
book_data = ingestor.data_load('documents/books')

research_data = ingestor.research_data_load('documents/research paper','research')
interview_data = ingestor.data_load('documents/interview')
article_data = ingestor.article_data_load('https://www.geeksforgeeks.org/nlp/advanced-natural-language-processing-interview-question/','article')

#chunking the book data

chunked_book_data = ingestor.book_research_chunking(book_data)
chunked_research_data = ingestor.book_research_chunking(research_data)
chunked_interview_data = ingestor.interview_web_chunking(interview_data)
chunked_article_data = ingestor.interview_web_chunking(article_data)

# adding embedded data to chromadb
embedding_book_data = ingestor.adding_data_to_chromadb(chunked_book_data)
embedding_research_data = ingestor.adding_data_to_chromadb(chunked_research_data)
embedding_interview_data = ingestor.adding_data_to_chromadb(chunked_interview_data)
embedding_article_data = ingestor.adding_data_to_chromadb(chunked_article_data)

# checking data quality report
#print(ingestion.data_quality_report('documents/books','book'))

#testing 4 retriveval quality test.

loaded_data = ingestor.other_data_load('evaluate_data.csv','csv')
chunk_data = ingestor.interview_web_chunking(loaded_data)
ingestor.adding_data_to_chromadb(chunk_data)
