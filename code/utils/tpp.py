import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from utils.config import *

def remove_stopwords(string):
    a = [i for i in string.split() if i not in stopwords.words('english')]
    
    return ' '.join(a)

def preprocess(text):
    text = text.lower() 
    text = text.strip()  
    text = re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    
    return text

def stem(row):
    snowballStemmer = SnowballStemmer("english")
    stemmedRow = [snowballStemmer.stem(word) for word in row.split()]
    
    return ' '.join(stemmedRow)

def process_row(row):
    row = preprocess(row)
    row = remove_stopwords(row)
    row = stem(row)
    
    return row

def run_tpp(data):
    data["clean_text"] = data[INPUT_NAME].apply(lambda x: process_row(x))
    
    return data