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

import pinecone
import streamlit as st

# load from local docs
def load_document(file_name):
  from langchain.document_loaders.csv_loader import CSVLoader
  loader = CSVLoader(file_name)
  return loader.load()


# create chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap
  )
  chunks = text_splitter.split_documents(data)
  return chunks


# compute embedding cost for a given set of chunks
def get_embedding_cost(chunks):
  import tiktoken
  enc = tiktoken.encoding_for_model('text-embedding-ada-002')
  total_tokens = sum([len(enc.encode(page.page_content)) for page in chunks])
  cost = (total_tokens / 1000 * 0.0004)
  return total_tokens, cost


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
def get_answers(vector_store, query, k=3):
  from langchain.chains import RetrievalQA
  from langchain.chat_models import ChatOpenAI
  
  # retriever makes it easy to combine documents with llms
  llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1.0)
  retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

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


if __name__ == '__main__':
  
  import os
  
  # load env vars
  from dotenv import load_dotenv, find_dotenv
  load_dotenv(find_dotenv(), override=True)
  
  st.image('./static/images/llm.jpeg')
  st.subheader('Ask a question')
  
  with st.sidebar:
    st.title('Influence LLM response')
    
    # get user inputs to influence the model
    dataset = st.file_uploader('Upload CSV Dataset', type=['csv'])
    chunk_size = st.number_input('Chunk Size', min_value=128, max_value=2056, value=512)
    chunk_overlap = st.number_input('Chunk Overlap', min_value=0, max_value=20, value=10)
    k = st.number_input('K', min_value=1, max_value=20, value=3)    
    upload_btn = st.button('Upload')
    
    # dataset uploaded & button clicked
    if upload_btn and dataset:
      st.spinner('Loading dataset...')
      
      # save dataset to a file in uploads folder
      bytes = dataset.read()
      dataset_file_name = os.path.join('./uploads/', dataset.name)
      with open(dataset_file_name, 'wb') as f:
        f.write(bytes)
    
      # load dataset
      f'loading dataset...'
      data = load_document(dataset_file_name)
      f'records: {len(data)}'
      st.divider()
    
      # create chunks
      f'creating chunks...'
      chunks = chunk_data(data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
      f'chunks: {len(chunks)}'
      st.divider()

      # get embeddings cost
      f'calculating tokens & embeddings cost...'
      tokens, cost = get_embedding_cost(chunks)
      f'tokens: {tokens}, cost: {cost:.6f}'
      st.divider()
      
      # create pinecone index and insert embeddings
      f'creating embeddings...'
      delete_pinecone_index('all')
      idx_name = dataset.name.replace('_', '-').replace('.csv', '')
      vector_store = insert_or_fetch_embeddings(idx_name, chunks)
      f'done'
      
      # success
      st.success('Dataset uploaded, chunked & embedded successfully')
