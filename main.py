import streamlit as st
import nltk
nltk.download('brown')
nltk.download('treebank')
nltk.download('universal_tagset')
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import InputLayer,Dense,Embedding,SimpleRNN,TimeDistributed
from nltk.tokenize import word_tokenize
from nltk.corpus import brown,treebank
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
import os

st.title("Parts of Speech Tagger")
text = st.text_area("Enter a word/sentence")
words = text.split()
if st.button("OK"):
    current_dir = os.path.dirname(__file__)
    file_path_model = os.path.join(current_dir, "pos_model.pkl")
    with open(file_path_model, "rb") as f:
        model = pickle.load(f)
    file_path_tk_x = os.path.join(current_dir, "tk_x.pkl")    
    with open(file_path_tk_x, "rb") as f:
        tk_x = pickle.load(f)
    file_path_tk_y = os.path.join(current_dir, "tk_y.pkl")    
    with open(file_path_tk_y, "rb") as f:
        tk_y = pickle.load(f)    


    te=[text]
    test=pad_sequences(tk_x.texts_to_sequences(te),maxlen=271,padding='post')
    tags = tk_y.sequences_to_texts((([np.argmax(model.predict(test)[0],axis=1)[np.argmax(model.predict(test)[0],axis=1)!=0].tolist()])))[0].split()
    
    for word,tag in zip(words,tags):
        st.subheader(f"{word} - {tag}")

else:
    pass            