import streamlit as st
import pickle
import nltk
nltk.data.path.append('./nltk_data/')
import string
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
import re

lemmatizing=WordNetLemmatizer()
def transform_text(text):
    text = text.lower()
    text=re.sub('[^a-zA-Z]',' ',text)
    text = nltk.word_tokenize(text)
    y = []
    text=[lemmatizing.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
    text=' '.join(text)
    y.append(text)
    y=' '.join(y)
    return y

bow = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
st.title("Sukku Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message to check with 98% accuracy")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = bow.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")