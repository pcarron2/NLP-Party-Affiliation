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

def saveToPickle(dict,name):
	pickle.dump(dict,open(name,"wb"))
	return

def loadFromPickle(filename):
	dictLoad=pickle.load(open(filename,"rb"))
	return copy.deepcopy(dictLoad)


if __name__ == '__main__':
	#Change the paths to match up with your corresponding path
	#The convote folder is the Congressional Speech data set made by Lillian Lee.
	destPath="/Users/nugthug/PycharmProjects/Party-Affiliation/PickleDicts/"
	trainPath="/Users/nugthug/Documents/cmpsci/585/f2016/Project/convote/data_stage_one/training_set/"
	testPath="/Users/nugthug/Documents/cmpsci/585/f2016/Project/convote/data_stage_one/test_set/"
	devPath="/Users/nugthug/Documents/cmpsci/585/f2016/Project/convote/data_stage_one/development_set/"
	trainFileNames=os.listdir(trainPath) 
	testFileNames=os.listdir(testPath)
	devFileNames=os.listdir(devPath)
	train_labels=np.array([])
	test_labels=np.array([])
	vectorizer=CountVectorizer()
	trainCorpus=[]
	for fn in trainFileNames:
		#Take in each file name from the training set directory
		#and use the 20th character of each file name as labels.
		#This character will either be a "D", "R", or "I".
		#These characters represent Democrat, Republican, or Independent
		#respectively. Each speech is then recorded into our training corpus.
		train_labels=np.append(train_labels,fn[19])
		f=open(os.path.join(trainPath,fn))
		trainCorpus.append(f.read())
		f.close()
	for fn in devFileNames:
		#Each of the development set directory is used to expand
		#the training set. This loop works in a similar manner as the
		#above loop.
		train_labels=np.append(train_labels,fn[19])
		f=open(os.path.join(devPath,fn))
		trainCorpus.append(f.read())
		f.close()
	testCorpus=[]
	print "Number of Documents in Training Set: "+str(len(trainFileNames))
	for fn in testFileNames:
		#This loop creates the labels for the test set
		#and reads in data for the test corpus. Works similarly
		#as the above for loops.
		test_labels=np.append(test_labels,fn[19])
		f=open(os.path.join(testPath,fn))
		testCorpus.append(f.read())
		f.close()
	#This pipeline holds the CountVectorizer, TF-IDF, and our classifier
	#as mentioned in the final report. In the pipeline the MultinomialNB
	#is initially set to alpha = 0 for our baseline case, as well as not 
	#using inverse document weighting.
	text_pipeline=Pipeline([('vect', CountVectorizer()),
							('tfidf', TfidfTransformer(use_idf=False)),
							('clf',  MultinomialNB(alpha=0))])
	text_pipeline=text_pipeline.fit(trainCorpus,train_labels)
	predicted=text_pipeline.predict(testCorpus)
	print "NB Accuracy: "+str(np.mean(predicted==test_labels))

	#Turn the training corpus into a dictionary of vocabulary of the training
	#corpus and into a frequency array that tells how frequent a term is.
	#
	wordFreqTrain=text_pipeline.named_steps['vect'].fit_transform(trainCorpus).toarray()
	wordNameDict=text_pipeline.named_steps['vect'].get_feature_names()
	wordFreqArray=np.sum(wordFreqTrain, axis=0)
	print "Woo"
	print wordFreqTrain
	print wordNameDict
	print wordFreqArray

	#Get basic statistics on our training corpus, such as total 
	#number of training tokens and vocabulary size. 
	#Also iterates through the top 100 and lowest 200 words in the 
	#training corpus.
	print "Number of training Tokens: "+str(sum(wordFreqArray))
	print "Size of training Vocabulary: "+str(len(wordNameDict))
	wordCountDict=dict(zip(wordNameDict,wordFreqArray))
	wordCountList=sorted(wordCountDict,key=wordCountDict.get, reverse=True)
	print "Top 100 words below: "
	for i in range(101):
		print str(wordCountList[i]) +" : "+ str(wordCountDict[str(wordCountList[i])])
	print "Lowest 200 words below: "
	for i in range(201):
		print str(wordCountList[-i])

	#Creates a Zipf's law graph in a log scaled form for our training
	#corpus. 
	length=len(wordCountList)
	rank=range(1,length+1) #Rank goes from 1 to however many words there are in the vocabulary.
	log_rank=[math.log(x) for x in rank] #The array of log rankings to plot on the x-axis.
	log_freq=[math.log(wordCountDict[word]) for word in wordCountList] #Array of log word frequencies plotted along the y-axis.
	fig = plt.figure()
	ax = plt.gca()
	ax.scatter( log_rank, log_freq, linewidth=2)
	plt.xlabel("log(rank)")
	plt.ylabel("log(frequency)")
	plt.title("Zipfs Law on Training Corpus")
	plt.savefig("zipfslaw.pdf")

	#This is where we start finding optimal hyper-parameters for MultinomialNB
	#as well as choosing a bigram language model versus a unigram language model
	#as well as determining whether or not to use inverse document frequency weighting.
	#Then we fit on the training set and evaluate on the test set. This is then
	#pickled to allow for faster reboots of this program.
	#Note that this program will use all your cores. In order to reduce this
	#please set n_jobs in GridSearchCV to however many cores you want the 
	#process to take.
	params={"vect__ngram_range":[(1,1),(1,2)],
			"tfidf__use_idf":(True,False),
			"clf__alpha":[0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,1]}
	crossValidationClf=GridSearchCV(text_pipeline,params,n_jobs=-1)
	crossValidationClf=crossValidationClf.fit(trainCorpus,train_labels)
	best_parameters, score, _ = max(crossValidationClf.grid_scores_ , key=lambda x: x[1])
	for param_name in sorted(params.keys()):
		print("%s: %r" % (param_name, best_parameters[param_name]))
	predicted=crossValidationClf.predict(testCorpus)
	print "NB CV Accuracy: "+str(score)
	print "NB CV Test Accuracy: "+str(np.mean(predicted==test_labels))
	saveToPickle(crossValidationClf.grid_scores_,destPath+"NBGSWDev.p")

	#This part of the program will use a SGD classifier instead of
	#a MultinomialNB like before. We similarly find optimal hyper-parameters,
	#albeit the SGD classifier has different hyper-parameters to optimize.
	#Then we train on the training data and predict on the test data in order
	#to evaulate the general accuracy. This is also pickled to reduce time for
	#in the case you need to run the program again. Same note again on the amount
	#of cores this program will use as before in the MultinomialNB.
	text_pipeline=Pipeline([('vect', CountVectorizer()),
							('tfidf', TfidfTransformer()),
							('clf',  SGDClassifier())])
	text_pipeline=text_pipeline.fit(trainCorpus,train_labels)
	params={"vect__ngram_range":[(1,1),(1,2)],
			"tfidf__use_idf":[True,False],
			"clf__alpha":[.00001,.0001,.0002,.0003,.001,.01,.1,1],
			"clf__loss":["hinge","log","perceptron"],
			"clf__penalty":["l1", "l2"]}
	crossValidationClf=GridSearchCV(text_pipeline,params,n_jobs=-1)
	crossValidationClf=crossValidationClf.fit(trainCorpus,train_labels)
	best_parameters, score, _ = max(crossValidationClf.grid_scores_ , key=lambda x: x[1])
	for param_name in sorted(params.keys()):
		print("%s: %r" % (param_name, best_parameters[param_name]))
	predicted=crossValidationClf.predict(testCorpus)
	print "SVM CV Accuracy: "+str(score)
	print "SVM CV Test Accuracy: "+str(np.mean(predicted==test_labels))
	saveToPickle(crossValidationClf.grid_scores_,destPath+"sdgClassifierGSWdevBig2.p")