# Modules
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

# Body
def clean(text):
    snowballStemmer = SnowballStemmer("english")
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [snowballStemmer.stem(word) for word in tokens]
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    
    return tokens

    
def gen_frequencies(tokens, save=False):
    return nltk.FreqDist(tokens)
    
def main(rawText):
    text = clean(rawText)
    freqs = gen_frequencies(text)
    print(freqs.most_common(5))
    
if __name__ == "__main__":
    rawText = "This is a Demo Text for NLP using NLTK. Full form of NLTK is Natural Language Toolkit"
    main(rawText)