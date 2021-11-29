import mmsdk
from mmsdk import mmdatasdk
import numpy as np
import nltk
from nltk.corpus import stopwords
import pickle

mydictLabels={'myfeaturesLabels':'cmumosi/CMU_MOSI_Opinion_Labels.csd'}
#mydictText = {'myfeaturesText':'cmumosi/CMU_MOSI_TimestampedWordVectors.csd'}
mydatasetLabels = mmdatasdk.mmdataset(mydictLabels)
#mydatasetText = mmdatasdk.mmdataset(mydictText)
#print(mydataset.computational_sequences['myfeatures'].data)


#Get text with labels 
totalSegments = 0
for key in mydatasetLabels.computational_sequences['myfeaturesLabels'].data.keys():
	totalSegments += len(mydatasetLabels.computational_sequences['myfeaturesLabels'].data[key]['features'])

textInput = np.zeros(totalSegments, dtype=object)
labelInput = np.zeros(totalSegments)
segmentCounter = 0
for key in mydatasetLabels.computational_sequences['myfeaturesLabels'].data.keys():
	textPath = 'Segmented/%s.annotprocessed' % (key)
	with open(textPath) as file: # Use file to refer to the file object
		text = file.read()
		text = text.replace("_DELIM_", "")
	text = text.split("\n")
	for segment in range(len(mydatasetLabels.computational_sequences['myfeaturesLabels'].data[key]['features'])):
		labelInput[segmentCounter] = mydatasetLabels.computational_sequences['myfeaturesLabels'].data[key]['features'][segment]
		text[segment] = ''.join([i for i in text[segment] if not i.isdigit()])
		textInput[segmentCounter] = text[segment]
		segmentCounter += 1



def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

def cleanup_labels(labels):
	newLabels = np.zeros(len(labels), dtype=object)
	for i, label in enumerate(labels): 
		if label >= 1:
			newLabels[i] = 'pos'
		elif label <= -1:
			newLabels[i] = 'neg'
		else:
			newLabels[i] = "neutral"
	return newLabels

def getWordsWithLabel(textInput, labelInput):
	words = []
	counter = 0
	for sentence in textInput:
		for word in sentence.split(" "):
			if word not in words:
				words.append((word, labelInput[counter]))
		counter += 1
	return words

def cleanUp(words):
	stop_words = set(stopwords.words('english'))
	words = [w for w in words if not w.lower() in stop_words]
	words = [w for w in words if not w.lower() in [".", ",", "(", ")", "\"", "-", "'", ":"]]
	return words

def getWords(labeledWords):
	return [x[0] for x in labeledWords]


# f = open('my_classifier.pickle', 'rb')
# classifier = pickle.load(f)
# f.close()
labelInput = cleanup_labels(labelInput)
labeledWords = getWordsWithLabel(textInput, labelInput)
words = getWords(labeledWords)
#Get list of each word in movie review plus the label 
# documents = [(list(words), category)
#               for category in movie_reviews.categories()
#               for fileid in movie_reviews.fileids(category)]

# #Get frequency of each word
all_words = nltk.FreqDist(w.lower() for w in words)

# # #Get list of the top 2000 most used words 
word_features = [i[0] for i in all_words.most_common(3000)]
word_features = cleanUp(word_features)

#For each review get a list with T/F values for each word in the top 2000 words 
#I guess this means that it gets points for not having 
#negative words or loses points for not having positve words?
featuresets = [(document_features(d), c) for (d,c) in labeledWords]

#Effectively run two epochs on data (couldn't figure out a better way to do this)
featuresets += featuresets

#Get train and test set. Then train model using natural language toolkit library 
train_set, test_set = featuresets[11397:], featuresets[:11397]
classifier = nltk.NaiveBayesClassifier.train(train_set)

#Print accuracy
print(nltk.classify.accuracy(classifier, test_set))

#Shows words that most informed the output
classifier.show_most_informative_features(5)
f = open('my_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()
