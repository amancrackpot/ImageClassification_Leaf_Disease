from io import BytesIO
import requests
import tensorflow as tf
import numpy as np
import PIL
import pathlib
import platform
import streamlit as st
STREAMLIT_THEME_BASE='light'

st.set_page_config(layout='centered')
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath


export_file_name = 'final_model.h5'
classes = ['Cassava Bacterial Blight (CBB)', 'Cassava Mosaic Disease (CMD)', 'Cassava Brown Streak Disease (CBSD)', 'Cassava Green Mottle (CGM)', 'Healthy']
path = pathlib.Path(__file__).parent

if 'learn' not in st.session_state :
    st.session_state.fin_model = tf.keras.models.load_model(path/'saved'/export_file_name)

def show_results(img):
    image_numpy = np.expand_dims(np.array(img), 0) #add batch dim
    outputs = st.session_state.fin_model(image_numpy, training=False).numpy()
    label = classes[np.argmax(outputs)]

    pred_probs = outputs.numpy()*100
    df = pd.DataFrame({'Label':classes,'Confidence':pred_probs}).set_index('Label')
    
    col1, col2 = st.beta_columns(2)
            
    with col1:
        st.subheader('Uploaded Image')
        st.image(img)
            
    with col2:   
        st.subheader('Analysis Report')
        st.table(df)
        st.info(f'Predicted Label : {label}')
    
padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)


st.title('Chest X-Ray Scan Classifier')
st.markdown('<hr>',unsafe_allow_html=True)
st.write('Find out if the lungs are healthy or not. Upload Chest X-Ray Scanned Image or Specify URL.')


st.sidebar.title('Configurations')
st.sidebar.write('')
st.sidebar.write('')
menu = ['Demo','Upload', 'URL']
choice = st.sidebar.selectbox("Select Image Source", menu)
st.sidebar.write('')
st.sidebar.write('')
cont = st.sidebar.container()
st.sidebar.write('')
st.sidebar.write('')
btn = st.sidebar.button('Analyze')

if choice == 'Upload':
    uploaded_file = cont.file_uploader("Upload an Image...", type=["jpg",'png','jpeg'])
    
    if btn and uploaded_file is not None:
        with st.spinner(text='Analyzing...'):
            try:
                img = PIL.Image.open(uploaded_file)
                show_results(img)
            except:
                st.error('Invalid File uploaded')

elif choice == 'URL':
    url = cont.text_input("Specify Image URL...")
        
    if btn and url is not '':
        with st.spinner(text='Analyzing...'):
            try:
                content = requests.get(url).content
                img = BytesIO(content)
                img = PIL.Image.open(img)
                show_results(img)
            except:
                st.error('URL specified is invalid')

else:
    cont.write('Runs demo on a sample Plant Image')
    url = 'https://drive.google.com/uc?export=download&id=1BMJobfXZAmPMSwvlGxf1mUYTleJBDBnx'
    if btn:
        with st.spinner(text='Analyzing...'):
            content = requests.get(url).content
            img = BytesIO(content)
            img = PIL.Image.open(img)
            show_results(img)

    
with st.expander("Dataset Link"):
    st.markdown('https://www.kaggle.com/c/cassava-leaf-disease-classification')
   
