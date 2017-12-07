# NLP-Party-Affiliation
Predicting party affiliation from speech

This project analyzes a US congressional speech corpus available at http://www.cs.cornell.edu/home/llee/data/convote/convote_v1.1.tar.gz


Abstract:
We attempt to classify a speaker’s party affiliation from their word usage by analyzing a corpus of congressional speeches. We investigate the impact of preprocessing techniques on cross-validation accuracy and test Multinomial Naive Bayes and Stochastic Gradient Descent classifiers. We perform 3-fold cross-validation on each classifier to find optimal hyper-parameters. We conclude that bigram language modeling with Tf-Idf weighting results in optimal preprocessing and Stochastic Gradient Descent classification with Hinge loss, l2 regularization, and a regularization constant of α = .0001 result in an optimal observed test accuracy of 74.9%.
