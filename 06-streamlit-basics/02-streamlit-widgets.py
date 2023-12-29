# run using: streamlit run 02-streamlit-widgets.py

import pandas as pd
import streamlit as st

st.title('Streamlit Widgets')
st.divider()

# text input
'Widget: Text Input'
name = st.text_input('Your name here')
if name:
  f'Hello {name}!'
st.divider()

# number input
'Widget: Number Input'
x = st.number_input('Enter a  number', min_value=1, max_value=99, step=1)
f'The current number is {x}'
st.divider()

# button
'Widget: Button'
clicked = st.button('Click me!')
if clicked:
  ':ghost:' * 3
st.divider()

# checkbox
'Widget: Checkbox'
agree = st.checkbox('I agree')
if agree:
  'Great, you agreed!'
st.divider()

'Widget: Checkbox (default checked)'
checked = st.checkbox('Continue', value=True)
if checked:
  ':+1:' * 5
st.divider()

'Widget: Show something if checkbox selected'
df = pd.DataFrame({
  'Name': ['Anne', 'Mario', 'Douglas'],
  'Age': [30, 25, 40]
})
if st.checkbox('Show data'):
    st.write(df)
st.divider()

# radio button
'Widget: Radio Button'
pets = ['cat', 'dog', 'fish', 'turtle']
pet = st.radio('Favorite pet', pets, index=2, key='your_pet')
st.write(f'Your favorite pet: {pet}')
st.write(f'Your favorite pet deom session state: {st.session_state.your_pet}')
st.divider()

# select boxes
'Widget: Select Box'
cities = ['London', 'Berlin', 'Paris', 'Madrid']
city = st.selectbox('Your city', cities, index=1)
st.write(f'Selcted city: {city}')
st.divider()

# slider
'Widget: Slider'
x = st.slider('x', value=1, min_value=1, max_value=50, step=3)
st.write(f'x is {x}')
st.divider()

# file uploader
'Widget: File Uploader'
uploaded_file = st.file_uploader('Upload a file', type=['txt', 'csv', 'xlsx'])
if uploaded_file:
  st.write(uploaded_file)
  if uploaded_file.type == 'text/plain':
    from io import StringIO
    stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
    string_data = stringio.read()
    st.write(string_data)
  elif uploaded_file.type == 'text/csv':
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    st.write(df)
  else:
    import pandas as pd
    df = pd.read_excel(uploaded_file)
    st.write(df)
st.divider()

# camera input
'Widget: Camera Input'
camera_photo = st.camera_input('Give a selfie')
if camera_photo:
  st.image(camera_photo)
st.divider()
