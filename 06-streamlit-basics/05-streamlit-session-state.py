# Run using: streamlit run 05-streamlit-session-state.py

import streamlit as st

st.title('Streamlit Session State')
st.divider()

# initialize state
if 'counter' not in st.session_state:
  st.session_state['counter'] = 0
    
button = st.button('Update state')
if button:
  st.session_state.counter += 1
st.divider()

'Session state'
st.write(st.session_state)
st.divider()
