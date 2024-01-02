### Getting hands dirty with AI
- Just playing around with LangChain, OpenAI, Pinecone & Streamlit
- To run the examples below:
  - setup virtualenv: https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/
  - cd to a directory and `pip install -r ../requirements.txt`
  - update the `.env` file in that folder with OpenAI's & Pinecone's api key
  - execute entire notebook or individual cells as needed
- Bullets below mention what parts are covered in each section  

### 01-langchain-basics
- OpenAI Setup
- OpenAI Text & Chat Models
- Prompt Templates
- Simple & Sequential Chains
- LangChain Agents

### 02-pinecone-basics
- Pinecone Setup
- Indexes (create, update, delete)
- Vectors (create, update, delete)

### 03-text-splitting-and-embedding
- Take a local doc, split into chunks, create embeddings & save them in pincecone
- Use a similarity search to get chunnks most simiar to user's query
- Feed the same chunks to LLM for a natural language response

### 04-app-custom-chatgpt
- Create three versions of a chatbot 
- One without memory
  - context not maintained between questions
  - each question is a standalone question
- One with memory
  - building on top of previous example
  - in-memory list of Q&A maintained after each question
  - for each new question, the in-memory list is passed to LLM
  - this help's LLM give contextual answers  
- One with persistance
  - building on top of previous example
  - sessions saved to json & can be loaded from disk
  - similar to ChatGPT interface

### 05-app-rag-netflix-engagement
- Example for Retrieval Augmented Generation (RAG)
  - gpt-3.5-turbo has a knowledge cutoff date of Jan 20222
  - Take the user engagement numbers from Netflix for Jan-Jun 2023
  - Feed them to LLM & ask questions
- No UI

### 06-streamlit-basics
- Widets (number & text inputs, buttons, checkboxes, sliders, radio buttons, file uploaders, progress bars)
- Layouts
- Session state & callbacks

### 07-dataset-generation
- Scraping IMDb top 100 movies of 2023
- Saving that as a csv dataset for next apps

### 08-app-chatgpt-user-uploads
- Similar to **05-app-rag-netflix-engagement** but with enhancements
- Users can upload any docs to feed to LLM
- Nice UI using streamlit with chat history rendered

### 09-app-text-summaries
- Summarizing using basic prompts
- Summarizing using prompt templates
- Summarizing using StuffDocumentsChain
- Summarizing using StuffDocumentsChain
- Summarizing using "map-reduce"
  - with default prompts
  - with custom prompts
- Summarizing using "refine"
  - with default prompts
  - with custom prompts
- Summarizing using LangchainAgents

### 10-app-chatgpt-using-streamlit-chat
- Simple UI showing chat interface using "streamlit-chat" instead of building of our interface
