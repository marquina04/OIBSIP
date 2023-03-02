import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

tfidf=pickle.load(open('vec.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

port_stem=PorterStemmer()


def text_processing(s):
    s=s.lower()
    s=nltk.word_tokenize(s)
    li=[]
    for i in s:
        if i.isalnum():
            li.append(i)
    s=li[:]
    li=[]
    for i in s:
        if i not in stopwords.words('english') and i not in string.punctuation:
            li.append(i)
    s=li[:]
    li=[]
    for i in s:
        li.append(port_stem.stem(i))


    return " ".join(li)

st.title("Email Spam Classifier")
email=st.text_area("Enter the email")
if st.button('Submit'):
    processed_email=text_processing(email)
    vector=tfidf.transform([processed_email])
    result=model.predict(vector)[0]
    if result==1:
        st.header("Spam Email")
    else:
        st.header("Not Spam Email")

