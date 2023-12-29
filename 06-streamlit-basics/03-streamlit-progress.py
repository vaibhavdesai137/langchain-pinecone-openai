# run using: streamlit run 03-streamlit-progress.py

import streamlit as st
import time

st.title('Streamlit Progress Bar')

'Starting computation ...'
latest_iteration = st.empty()

text = 'Operation in progress. Please wait...'
my_bar = st.progress(0, text=text)
time.sleep(2)

# increase progress bar every 0.1 seconds
for i in range(100):
  my_bar.progress(i+1)
  latest_iteration.text(f'Iteration {i+1}')
  time.sleep(0.1)

'Done :+1:'
