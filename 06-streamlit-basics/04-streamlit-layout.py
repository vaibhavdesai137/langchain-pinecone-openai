# Run using: streamlit run 04-streamlit-layout.py

import streamlit as st
import random

st.title('Streamlit Layout')
st.divider()

# sidebar
st.sidebar.title('This is a sidebar')
st.sidebar.divider()
name = st.sidebar.text_input('Whats your name?')
if name:
  st.sidebar.write(f'Hello {name}!')

# 2 column layout
'2 Column Layout Example'
left_column, right_column = st.columns(2)
with left_column:
  data = [random.random() for _ in range(100)]
  'Col 1'
  st.line_chart(data)
with right_column:
  data = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
  'Col 2'
  data[:4]
st.divider()

# 3 column layout
'3 Column Layout Example'
col1, col2, col3 = st.columns([0.4, 0.2, 0.4])

with col1:
  'Col 1 - a cat'
  st.image('https://static.streamlit.io/examples/cat.jpg')
with col3:
  'Col 3 - a dog'
  st.image('https://static.streamlit.io/examples/dog.jpg')
st.divider()

# expander
'Accordion Example'
with st.expander('Expand me'):
  st.bar_chart({'Data': [random.randint(2, 10) for _ in range(25)]})
  st.write('Random images')
  st.image('https://static.streamlit.io/examples/cat.jpg')
  st.image('https://static.streamlit.io/examples/dog.jpg')
