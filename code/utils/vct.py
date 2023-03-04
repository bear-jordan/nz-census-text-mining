from sklearn.feature_extraction.text import TfidfVectorizer
from utils.config import *

def run_vct(data, newData):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data["clean_text"])
    Xnew = vectorizer.transform(newData["clean_text"])
    
    return (X, Xnew)