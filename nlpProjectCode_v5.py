import os
from collections import defaultdict
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
	#Change the paths to match up with your corresponding path
	#Works similarly to the main code for this portion of the code.
	#Make sure to download the Congressional Speech data set made by
	#Lillian Lee from Cornell. 
	#In this portion of the code we separate the training set directory to 
	#have a democrat set and a republican set.
	#This will be used for purely analytical reasons to find terms comparing
	#the Democrat versus the Republican party.
	destPath="PICKLE DICTIONARY PATH"
	trainPath="TRAINING SET PATH"
	testPath="TEST SET PATH"
	devPath="DEV SET PATH"
	trainFileNames=os.listdir(trainPath)
	testFileNames=os.listdir(testPath)
	devFileNames=os.listdir(devPath)
	train_labels=np.array([])
	demo_label = np.array([])
	repub_label = np.array([])
	test_labels=np.array([])
	test_demo_label = np.array([])
	test_repub_label = np.array([])
	vectorizer=CountVectorizer()
	trainCorpus=[]
	trainDemo = []
	trainRepub = []
	for fn in trainFileNames:
		#Instead of adding the data into the training corpus, like in the main
		#code we add them to a democrat data set and a republican data set from
		#the training set path in the Congressional Speech data set's directory.
		f=open(os.path.join(trainPath,fn))
		if fn[19] == 'D':
			demo_label=np.append(demo_label,fn[19])
			trainDemo.append(f.read())
		if fn[19] == 'R':
			repub_label = np.append(repub_label,fn[19])
			trainRepub.append(f.read())
		f.close()
	for fn in devFileNames:
		#Similarly we add it to the democratic data set and the republican data set
		#that we added to before. Except in this loop we take data from the development
		#set directory.
		f=open(os.path.join(devPath,fn))
		if fn[19] == 'D':
			demo_label=np.append(demo_label,fn[19])
			trainDemo.append(f.read())
		if fn[19] == 'R':
			repub_label = np.append(repub_label,fn[19])
			trainRepub.append(f.read())
		f.close()
	testCorpus=[]
	testDemo = []
	testRepub = []
	print "Number of Documents in Training Set: "+str(len(trainFileNames))
	for fn in testFileNames:
		#Append additional data into the test data sets.
		f=open(os.path.join(testPath,fn))
		if fn[19] == 'D':
			test_demo_label=np.append(test_demo_label,fn[19])
			testDemo.append(f.read())
		if fn[19] == 'R':
			test_repub_label = np.append(test_repub_label,fn[19])
			testRepub.append(f.read())
		f.close()

	#In this part of the code we make a pipeline, though we only use the CountVectorizer
	#in this case. This portion of the code will focus on the Democratic party. 
	#First the code will fit and transform the democractic data set we appended before.
	#Then convert it to a format we can analyze and find bigrams within the dataset that
	#contain keywords we're looking for, such as energy, education, and healthcare.
	#We also conveniently provide a count to also print out if necessary.
	text_pipeline=Pipeline([('vect', CountVectorizer()),
							('tfidf', TfidfTransformer(use_idf=False)),
							('clf',  MultinomialNB(alpha=0))])

	vecti = CountVectorizer(ngram_range=(2, 2))
	freqTrain= vecti.fit_transform(trainDemo).toarray()
	nameDict = vecti.get_feature_names()
	freqArray = np.sum(freqTrain, axis=0)
	countDict = dict(zip(nameDict, freqArray))
	energyDict = {}
	educationDict = {}
	healthDict = {}
	print "Intriguing Bigrams Democrats: "
	for x in countDict:
		if "energy" in x:
			print x
			if x not in energyDict:
				energyDict[x] = 1
			else:
				energyDict[x] += 1
		elif "education" in x:
			print x
			if x not in educationDict:
				educationDict[x] = 1
			else:
				educationDict[x] += 1
		elif "healthcare" in x:
			print x
			if x not in healthDict:
				healthDict[x] = 1
			else:
				healthDict[x] += 1

	#Still focusing on the Democratic party we create a dictionary of word counts
	#in this Democrat data set and find basic statistics of it. Then we print out the
	#top 200 words for the democrat and the lowest 200 words in terms of frequencies for the rankings. 
	wordFreqTrain=text_pipeline.named_steps['vect'].fit_transform(trainDemo).toarray()
	wordNameDict=text_pipeline.named_steps['vect'].get_feature_names()
	wordFreqArray=np.sum(wordFreqTrain, axis=0)

	print "Number of training Tokens: "+str(sum(wordFreqArray))
	print "Size of training Vocabulary: "+str(len(wordNameDict))
	wordCountDict=dict(zip(wordNameDict,wordFreqArray))
	wordCountList=sorted(wordCountDict,key=wordCountDict.get, reverse=True)
	demoLib = wordCountList[:200] #Top 200 words of the Democrat data set.
	print "Top 200 Democrat words below: "
	for i in range(201):
		print str(wordCountList[i]) +" : "+ str(wordCountDict[str(wordCountList[i])])
	print "Lowest 200 Democrat words below: "
	for i in range(1, 201):
		print str(wordCountList[-i])

	#This part of the code will now do the same as above, except for the 
	#Republican party. Except that we will print the bigrams in the Republican
	#data set that isn't in the Democrat data set to make it easier to find 
	#differences between the data set.

	text_pipeline=Pipeline([('vect', CountVectorizer()),
							('tfidf', TfidfTransformer(use_idf=False)),
							('clf',  MultinomialNB(alpha=0))])

	vecti = CountVectorizer(ngram_range=(2, 2))
	freqTrain= vecti.fit_transform(trainRepub).toarray()
	nameDict = vecti.get_feature_names()
	freqArray = np.sum(freqTrain, axis=0)
	countDict = dict(zip(nameDict, freqArray))
	print "Intriguing Bigrams Republican: "
	for x in countDict:
		if "energy" in x:
			if x not in energyDict:
				print x
		elif "education" in x:
			if x not in educationDict:
				print x
		elif "healthcare" in x:
			if x not in healthDict:
				print x

	wordFreqTrain=text_pipeline.named_steps['vect'].fit_transform(trainRepub).toarray()
	wordNameDict=text_pipeline.named_steps['vect'].get_feature_names()
	wordFreqArray=np.sum(wordFreqTrain, axis=0)

	print "Number of training Tokens: "+str(sum(wordFreqArray))
	print "Size of training Vocabulary: "+str(len(wordNameDict))
	wordCountDict=dict(zip(wordNameDict,wordFreqArray))
	wordCountList=sorted(wordCountDict,key=wordCountDict.get, reverse=True)
	repubLib = wordCountList[:200]
	print "Top 200 Republican words below: "
	for i in range(201):
		print str(wordCountList[i]) +" : "+ str(wordCountDict[str(wordCountList[i])])
	print "Lowest 200 Republican words below: "
	for i in range(1, 201):
		print str(wordCountList[-i])

	#In this part of the code we analyze the top 200 words of each poltical party
	#and compare which words they have in common and which words they dont.
	finalLib = [] #Words that Democrat and Republican have in common
	notIn = [] #Words they do not have in common.
	for d in demoLib:
		for r in repubLib:
			if d==r:
				finalLib.append(str(d))
			if d!=r:
				notIn.append((str(d), str(r)))
	print "How many frequent word in common: "+str(len(finalLib))
	print "Difference in frequent words: "
	seen = [] #In order to not print a word multiple times we keep track of the words we've seen.
	for d, r in notIn:
		#Print words they do not have in common.
		#Check to make sure none of these words from both parties are not in
		#the list of words they have in common.
		if d not in finalLib and d not in seen:
			print "Democrat Word: "+d
			seen.append(d)
		if r not in finalLib and r not in seen:
			print "Republican word: "+r
			seen.append(r)
	print "Republican and Democrat In common words:"
	for x in finalLib:
		#Print words they have in common.
		print x
	
