# run using: streamlit run 01-streamlit-write.py

# Displaying data on the screen
# 1. st.write()
# 2. Magic. Any python variable or expression written on a line by itself will be printed out in the app. No need to use st.write().

import streamlit as st
import pandas as pd

# page title
st.title('Hello Streamlit World! :+1:')
st.divider()

# some random data to render
list = [1, 2, 3]
dict = {'a': 1, 'b': 2, 'c': 3}
df = pd.DataFrame({
  'first_column': [1, 2, 3, 4],
  'second_column': [10, 20, 30, 40]
})

# ----- using st.write() -----
st.write('Displaying using st.write() :smile:')
st.write(list)
st.write(dict)
st.write(df)
st.divider()

# ----- using magic -----
'Displaying using magic :smile:'
list
dict
df
