import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

stopwords = stopwords.words("english")
snowballStemmer = SnowballStemmer("english")

text = "This is a Demo Text for NLP using NLTK. Full form of NLTK is Natural Language Toolkit"
lowerText = text.lower()
wordTokens = nltk.word_tokenize(lowerText)
stemmedTokens = [snowballStemmer.stem(word) for word in wordTokens]
nonStopWords = [word for word in stemmedTokens if word not in stopwords]
frequencies = nltk.FreqDist(nonStopWords)

print(frequencies.most_common(5))
