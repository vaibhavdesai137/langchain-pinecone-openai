# ---------------
# **Retrieval Augmented Generation**
# - Retrieval Augmented Generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources
# - It is also a way to feed new/updated data into LLMs
# - For eg: kjnowledge cut off date for gpt-3.5-turbo is Jan 2022 so it won't know anything after that
# - https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/
# ---------------
# **Project Setup**
# - Same as 05-app-rag-netflix-engagement
# - But instead on a self generated dataset from imdb bout top movies in 2023
# - A nice UI using Streamlit to provide a ChatGPT like interface
# ---------------

import os
import pinecone


# load from local docs
def load_movies_csv():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  from langchain.document_loaders.csv_loader import CSVLoader
  dataset_path = current_dir + '/dataset/imdb_top_movies_2023.csv'
  loader = CSVLoader(dataset_path)
  return loader.load()


# create chunks
def chunk_data(data, chunk_size=256):
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = 10
  )
  chunks = text_splitter.split_documents(data)
  return chunks


# compute embedding cost for a given set of chunks
def get_embedding_cost(chunks):
  import tiktoken
  enc = tiktoken.encoding_for_model('text-embedding-ada-002')
  total_tokens = sum([len(enc.encode(page.page_content)) for page in chunks])
  cost = (total_tokens / 1000 * 0.0004)
  print(f'total tokens: {total_tokens}')
  return cost


# delete a given pinecone index or all indexes
def delete_pinecone_index(idx_name='all'):
  import os
  import pinecone
  pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENV")
  )  
  if idx_name == 'all':
    for index in pinecone.list_indexes():
      pinecone.delete_index(index)
  else:
    pinecone.delete_index(idx_name)


# create a pinecone index and insert embeddings or 
# fetch existing embeddings if index already exists
def insert_or_fetch_embeddings(idx_name, chunks):
  import os
  import pinecone
  from langchain.vectorstores import Pinecone
  from langchain.embeddings import OpenAIEmbeddings
  
  embeddings = OpenAIEmbeddings()
  pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENV")
  )
  
  if idx_name in pinecone.list_indexes():
    print(f'index: {idx_name} already exist, loading embeddings')
    vector_store = Pinecone.from_existing_index(idx_name, embeddings)
  else:
    print(f'creating index {idx_name} and embeddings')
    pinecone.create_index(name=idx_name, dimension=1536, metric='cosine')
    vector_store = Pinecone.from_documents(chunks, embeddings, index_name = idx_name)
    
  return vector_store


# get chunks from vector store based on similarity index
# pass those chunks to LLM to get answers in natural language
def get_answers(vector_store, query):
  from langchain.chains import RetrievalQA
  from langchain.chat_models import ChatOpenAI
  
  # retriever makes it easy to combine documents with llms
  llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1.0)
  retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})

  # create a chain that combines llm & retriever
  # chain_type='stuff' is a predefined chain that uses all text from documents
  chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type='stuff')
  resp = chain.run(query)
  return resp


# same as above but with memory i.e context awareness
def get_answers_with_memory(vector_store, query, chat_history=[]):
  from langchain.chat_models import ChatOpenAI
  from langchain.chains import ConversationalRetrievalChain
  
  llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
  retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
  
  crc = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
  resp = crc({'question': query, 'chat_history': chat_history})
  chat_history.append((query, resp['answer']))
  return resp, chat_history


# --------------------------------------------------

# load env vars
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# load dataset
data = load_movies_csv()
print(f'total records: {len(data)}')
  
# create chunks
chunks = chunk_data(data)
print(f'total chunks: {len(chunks)}')

# get embedding cost
cost = get_embedding_cost(chunks)
print(f'embedding cost: {cost:.6f}')

# create pinecone index and insert embeddings
delete_pinecone_index('all')
idx_name = 'imdb-top-movies-2023'
vector_store = insert_or_fetch_embeddings(idx_name, chunks)

while True:
  
  query = input('your prompt: ')
  
  # break if user wants to quit
  if query in ['quit', 'exit', 'bye', 'q']:
    print('Goodbye!')
    break
  
  # else ask the model for a response
  resp = get_answers(vector_store, query)
  print(f'prompt: {query}')
  print(f'resp: {resp}')
  print('_' * 50)
