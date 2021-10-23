import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import random

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

def cleanUp(words):
	stop_words = set(stopwords.words('english'))
	words = [w for w in words if not w.lower() in stop_words]
	words = [w for w in words if not w.lower() in [".", ",", "(", ")", "\"", "-", "'", ":"]]
	return words

#Get list of each word in movie review plus the label 
documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

#Shuffle the documents
random.shuffle(documents)

#Get frequency of each word
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())

#Get list of the top 2000 most used words 
word_features = [i[0] for i in all_words.most_common(3000)]
word_features = cleanUp(word_features)


#For each review get a list with T/F values for each word in the top 2000 words 
#I guess this means that it gets points for not having 
#negative words or loses points for not having positve words?
featuresets = [(document_features(d), c) for (d,c) in documents]

#Effectively run two epochs on data (couldn't figure out a better way to do this)
featuresets += featuresets

#Get train and test set. Then train model using natural language toolkit library 
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

#Print accuracy
print(nltk.classify.accuracy(classifier, test_set))

#Shows words that most informed the output
classifier.show_most_informative_features(5)
