import pickle
import string
import streamlit as st
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_mail(mail):
    mail=mail.lower() #lower case mai convert krne ke liye
    mail=nltk.word_tokenize(mail)# list mai convert krne ke liye
    y=[]
    for i in mail:
        if i.isalnum(): # function used use to check it is alphabet/number.
            y.append(i)
    mail=y[:] # ??
    y.clear()
    
    for i in mail:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    mail=y[:]
    y.clear()
    
    for i in mail:
        y.append(ps.stem(i))
    return " ".join(y) #string return ho rhi hai

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb')) 

st.title("Email Spam Classifier")

input_mail = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_mail = transform_mail(input_mail)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_mail])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.markdown("<h1 style='color: red;'>This is a Spam mail.</h1>", unsafe_allow_html=True)
    else:
       st.markdown("<h1 style='color: green;'>This is Not a Spam mail.</h1>", unsafe_allow_html=True)