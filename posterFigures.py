import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
import math
'''
This is the code we used to create our exhibits
We loaded our gridsearchcv files from pickle so we
could perfect our plots and not wait for new runs each time.

'''
def loadFromPickle(filename):
	dictLoad=pickle.load(open(filename,"rb"))
	return copy.deepcopy(dictLoad)

destPath='Figures/'
picklePath='PickleDicts/'



def plotSVMgridClose(svmGrid):
    nGrams=[(1,1),(1,2)]
    tdIdf=['True','False']
    alpha=np.arange(0.00005,.0009,.00005)
    loss=["hinge","log","perceptron"]
    penalty=["none","l1","l2"]
    scores=[feat[1] for feat in svmGrid]
    newMatrix=np.zeros((len(svmGrid),6),dtype=object)
    for i in range(len(svmGrid)):
        newMatrix[i,0]=svmGrid[i][1]
        newMatrix[i,1]=svmGrid[i][0]["vect__ngram_range"]
        newMatrix[i,2]=svmGrid[i][0]["tfidf__use_idf"]
        newMatrix[i,3]=svmGrid[i][0]["clf__alpha"]
        newMatrix[i,4]=svmGrid[i][0]["clf__loss"]
        newMatrix[i,5]=svmGrid[i][0]["clf__penalty"]
    metDict={}
    for met in loss:
        scoreArray=[]
        alphaList=[]
        for i in range(len(svmGrid)):
            if newMatrix[i,1]==(1, 2) and newMatrix[i,2]==True and newMatrix[i,5]=="none" and newMatrix[i,4]==met:
                scoreArray.append(newMatrix[i,0])
                alphaList.append(newMatrix[i,3])
        metDict[met]=[scoreArray,alphaList]
    typeDict={"hinge":'b-',"log":'g-',"perceptron":'r-'}
    plt.figure(1,figsize=(12,8))
    for met in metDict.keys():
        #print met, metDict[met]
        plt.plot(metDict[met][1],metDict[met][0],typeDict[met], linewidth=3)
    #plt.plot(kList,testErrList,'sb-', linewidth=3)
    plt.grid(True) #Turn the grid on
    plt.ylabel("Crossvalidation Accuracy") #Y-axis label
    plt.xlabel("Alpha") #X-axis label
    plt.title("Bigram Tf-Idf Log Hinge and Perceptrion Loss by Log Alpha") #Plot title
    plt.xlim(0,.00095) #set x axis range
    plt.ylim(.65,.72) #Set yaxis range
    plt.legend(["SVM","Logistic Regrssion","Perceptron"],loc="best")
    #Make sure labels and titles are inside plot area
    plt.tight_layout()
    #Save the chart
    plt.savefig(destPath+"FiguresSVM_line_plot.png")
    plt.show()
    plt.clf()


def plotSVMgridBig(svmGrid):
    nGrams=[(1,1),(1,2)]
    tdIdf=['True','False']
    #alpha=np.arange(0.00005,.0009,.00005)
    loss=["hinge","log","perceptron"]
    penalty=["none","l1","l2"]
    scores=[feat[1] for feat in svmGrid]
    newMatrix=np.zeros((len(svmGrid),6),dtype=object)
    for i in range(len(svmGrid)):
        newMatrix[i,0]=svmGrid[i][1]
        newMatrix[i,1]=svmGrid[i][0]["vect__ngram_range"]
        newMatrix[i,2]=svmGrid[i][0]["tfidf__use_idf"]
        newMatrix[i,3]=svmGrid[i][0]["clf__alpha"]
        newMatrix[i,4]=svmGrid[i][0]["clf__loss"]
        newMatrix[i,5]=svmGrid[i][0]["clf__penalty"]
    metDict={}
    for met in loss:
        scoreArray=[]
        alphaList=[]
        for i in range(len(svmGrid)):
            if newMatrix[i,1]==(1, 2) and newMatrix[i,2]==True and newMatrix[i,5]=="l2" and newMatrix[i,4]==met:
                scoreArray.append(newMatrix[i,0])
                alphaList.append(newMatrix[i,3])
        metDict[met]=[scoreArray,alphaList]
    typeDict={"hinge":'b-',"log":'g-',"perceptron":'r-'}
    plt.figure(1,figsize=(12,8))
    for met in metDict.keys():
        #print met, metDict[met]
        plt.plot(metDict[met][1],metDict[met][0],typeDict[met], linewidth=3)
    #plt.plot(kList,testErrList,'sb-', linewidth=3)
    plt.grid(True) #Turn the grid on
    plt.ylabel("Crossvalidation Accuracy") #Y-axis label
    plt.xlabel("Alpha") #X-axis label
    plt.title("Bigram Tf-Idf L2 Penalty SVM LR and Perceptron CV Scores by Alpha") #Plot title
    plt.xlim(0.0001,.003) #set x axis range
    plt.ylim(.6,.75) #Set yaxis range
    plt.legend(["SVM","Logistic Regrssion","Perceptron"],loc="best")
    #Make sure labels and titles are inside plot area
    plt.tight_layout()
    #Save the chart
    plt.savefig(destPath+"FiguresSVM_line_plot_bigalpha.png")
    plt.show()
    plt.clf()

def plotSVMgridTdidf(svmGrid):
    nGrams=[(1,1),(1,2)]
    tdIdf=['True','False']
    #alpha=np.arange(0.00005,.0009,.00005)
    loss=["hinge","log","perceptron"]
    penalty=["none","l1","l2"]
    scores=[feat[1] for feat in svmGrid]
    newMatrix=np.zeros((len(svmGrid),6),dtype=object)
    for i in range(len(svmGrid)):
        newMatrix[i,0]=svmGrid[i][1]
        newMatrix[i,1]=svmGrid[i][0]["vect__ngram_range"]
        newMatrix[i,2]=svmGrid[i][0]["tfidf__use_idf"]
        newMatrix[i,3]=svmGrid[i][0]["clf__alpha"]
        newMatrix[i,4]=svmGrid[i][0]["clf__loss"]
        newMatrix[i,5]=svmGrid[i][0]["clf__penalty"]
    metDict={}
    for met in loss:
        scoreArray=[]
        tdIdfList=[]
        for i in range(len(svmGrid)):
            if newMatrix[i,1]==(1, 2) and newMatrix[i,3]==0.0001 and newMatrix[i,5]=="l2" and newMatrix[i,4]==met:
                scoreArray.append(newMatrix[i,0])
                tdIdfList.append(newMatrix[i,2])
        metDict[met]=[scoreArray,tdIdfList]
    typeDict={"hinge":'b',"log":'g',"perceptron":'r'}

    #fig, ax=plt.subplots(figsize=(10,5))
    #for met in metDict.keys():
        #print met, metDict[met]
    labelDict={"hinge":"SVM","log":"Logistic Regression","perceptron":"Perceptron"}
    print str(labelDict["hinge"])
    for met in metDict.keys():
        inds=np.arange(2)
        labels=["No Tf-Idf","Yes Tf-Idf"]
        plt.figure(1,figsize=(7,5))
        plt.bar(metDict[met][1],metDict[met][0],align='center',color=typeDict[met])
    #plt.plot(kList,testErrList,'sb-', linewidth=3)
        plt.grid(True) #Turn the grid on
        plt.ylabel("Crossvalidation Accuracy") #Y-axis label
        plt.xlabel("Tf-Idf") #X-axis label
        plt.title("Bigram L2 Penalty Alpha=.0001 CV Scores by Tf-Idf") #Plot title
        plt.xlim(-.5,1.5) #set x axis range
        plt.ylim(.5,.8) #Set yaxis range
        plt.gca().set_xticks(inds)
        plt.gca().set_xticklabels(labels)
        plt.legend((labelDict[met],),loc="best")
    #Make sure labels and titles are inside plot area
        plt.tight_layout()
    #Save the chart
        plt.savefig(destPath+"FiguresSVM_TFIDF_on_off_"+met+".png")
        plt.show()
        plt.clf()


def plotSVMgridBigram(svmGrid):
    nGrams=[(1,1),(1,2)]
    tdIdf=['True','False']
    #alpha=np.arange(0.00005,.0009,.00005)
    loss=["hinge","log","perceptron"]
    penalty=["none","l1","l2"]
    scores=[feat[1] for feat in svmGrid]
    newMatrix=np.zeros((len(svmGrid),6),dtype=object)
    for i in range(len(svmGrid)):
        newMatrix[i,0]=svmGrid[i][1]
        newMatrix[i,1]=svmGrid[i][0]["vect__ngram_range"]
        newMatrix[i,2]=svmGrid[i][0]["tfidf__use_idf"]
        newMatrix[i,3]=svmGrid[i][0]["clf__alpha"]
        newMatrix[i,4]=svmGrid[i][0]["clf__loss"]
        newMatrix[i,5]=svmGrid[i][0]["clf__penalty"]
    metDict={}
    for met in loss:
        scoreArray=[]
        bigramList=[]
        for i in range(len(svmGrid)):
            if newMatrix[i,2]==True and newMatrix[i,3]==0.0001 and newMatrix[i,5]=="l2" and newMatrix[i,4]==met:
                scoreArray.append(newMatrix[i,0])
                bigramList.append(newMatrix[i,1])
        metDict[met]=[scoreArray,bigramList]
    typeDict={"hinge":'b',"log":'g',"perceptron":'r'}
    #fig, ax=plt.subplots(figsize=(10,5))
    #for met in metDict.keys():
        #print met, metDict[met]
    labelDict={"hinge":"SVM","log":"Logistic Regression","perceptron":"Perceptron"}
    #print str(labelDict["hinge"])
    for met in metDict.keys():
        inds=np.arange(2)
        labels=["Unigram","Bigram"]
        plt.figure(1,figsize=(7,5))
        plt.bar(inds,metDict[met][0],align='center',color=typeDict[met])
    #plt.plot(kList,testErrList,'sb-', linewidth=3)
        plt.grid(True) #Turn the grid on
        plt.ylabel("Crossvalidation Accuracy") #Y-axis label
        plt.xlabel("Language Model") #X-axis label
        plt.title("L2 Penalty Alpha=.0001 with Tf-Idf \n CV Scores by Language Model") #Plot title
        plt.xlim(-.5,1.5) #set x axis range
        plt.ylim(.5,.8) #Set yaxis range
        plt.gca().set_xticks(inds)
        plt.gca().set_xticklabels(labels)
        plt.legend((labelDict[met],))
    #Make sure labels and titles are inside plot area
        plt.tight_layout()
    #Save the chart
        plt.savefig(destPath+"FiguresSVM_bigram_"+met+".png")
        plt.show()
        plt.clf()



svmGrid=loadFromPickle(picklePath+"sdgClassifierGSWdev.p")
plotSVMgridClose(svmGrid)
#plotSVMgridBig(svmGrid)

svmGridbig=loadFromPickle(picklePath+'sdgClassifierGSWdevBig2.p')
plotSVMgridBig(svmGridbig)
#plotSVMgridBigram(svmGridbig)
#plotSVMgridTdidf(svmGridbig)
#plotSVMgridClose(svmGridbig)

svmGridbig=loadFromPickle(picklePath+'sdgClassifierGSWdevBig.p')
plotSVMgridBigram(svmGridbig)
plotSVMgridTdidf(svmGridbig)
#plotSVMgridClose(svmGridbig)