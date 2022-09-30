import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import pickle
from datetime import datetime
import plotly.express as px 

# load model with joblib
pipe_lr = joblib.load(open('models/Emotion_Classifier_pipe_lr.pkl', 'rb'))

# load model with pickle

# pickle_in = open('models/pipe_lr_pickle.pkl', 'rb')
# classifier = pickle.load(pickle_in)

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {
    "anger" : "😠", 
    "disgust" : "🤮", 
    "fear" : "😨😱", 
    "happy" : "🤗", 
    "joy" : "😂", 
    "neutral" : "😐", 
    "sad" : "😔", 
    "sadness" : "😔", 
    "shame" : "😳", 
    "surprise" : "😮"
    }

def main():
    st.title('Emotions Classifier')
    menu = ['Home', 'Monitor', 'About']
    choice = st.sidebar.selectbox('Menu', menu)
    
    if choice == 'Home':
        st.subheader('Home-Emotion in Text')
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area(placeholder='Text Here', label='Type down below in text area...')
            submit_text = st.form_submit_button(label='Submit')
            if submit_text:
                col1, col2 = st.columns(2)
                # apply function here
                prediction = predict_emotions(raw_text)
                probability = get_prediction_proba(raw_text)
                
                with col1:
                    st.success('Original Text')
                    st.write(raw_text)
                    st.success('Prediction')
                    emoji_icon = emotions_emoji_dict[prediction]
                    st.write('{}:{}'.format(prediction, emoji_icon))
                    st.write(
                        'Confidence Level:  {}'.format(
                            np.max(probability)
                                            )
                        )
                    
                with col2:
                    st.success('Prediction Probability')
                    st.write(probability)
                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    
                    st.write(proba_df.T)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ['emotions', 'probability']
                    
                    fig = alt.Chart(proba_df_clean).mark_bar().encode(
                        x='emotions', y='probability', color='emotions')
                
                    st.altair_chart(fig, use_container_width=True)
            
    elif choice == 'Monitor':
        st.subheader('Monitor App')

    else:
        st.subheader('About')
        st.write('This App is all about taking text and classifying it\'s emotion such as (happy, sad, fear.....)')
        st.write('Write text to classify.......')

if __name__ == '__main__':
    main()