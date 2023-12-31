# first install required packages using "pip install -r requirements.txt"
# then run using: streamlit run app.py
# 
# # ---------------
# - Simple ChatGPT like interface using Streamlit
# - Uses streamlit-chat package instead of building our own UI from scratch
# ---------------

import os
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

# load env vars & init chat model
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)

# clear history when a new dataset is uploaded
def clear_history():
  if 'history' in st.session_state:
    del st.session_state['history']

# page settings
st.set_page_config(
  page_title="Your Friendly Assitant",
  page_icon=":Laptop:"
)
st.subheader('Your Friendly Assitant')

# store chat history in 'messages' key in session state
if 'messages' not in st.session_state:
  st.session_state.messages = []
  
with st.sidebar:
  # allow users to set system role once
  # and ask questions as many times as they want
  system_msg = st.text_input('System Role')
  user_msg = st.text_input('Ask Me Something')

  # system msg can be set just once per session
  if system_msg:
    if not any(isinstance(x, SystemMessage) for x in st.session_state.messages):
      st.session_state.messages.append(SystemMessage(content=system_msg))
    
  # user msg can be an inifinite list
  if user_msg:
    # add user's new question to session state
    st.session_state.messages.append(HumanMessage(content=user_msg))  

    # get response from llm & save that to state as well
    with st.spinner('Thinking...'):    
      # give entire history to llm
      # this will have user's latest question as well
      resp = llm(st.session_state.messages)
      st.session_state.messages.append(AIMessage(content=resp.content))

# show all messages on UI from session state
# each widget needs a unique key so set that to i
for i, msg in enumerate(st.session_state.messages):
  if isinstance(msg, HumanMessage):
    message(msg.content, is_user=True, key=i)
  elif isinstance(msg, AIMessage):
    message(msg.content, is_user=False, key=i)
  else:
    print('Ignoring system message')
