#!/usr/bin/python
import re
import sys
from pprint import pprint
import operator

from nltk.corpus import gutenberg
from nltk.corpus import stopwords
from nltk.classify.svm import SvmClassifier, map_instance_to_svm, map_features_to_svm
from nltk.tokenize import word_tokenize
from nltk.classify import accuracy
from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

from pylab import *
import matplotlib.pyplot as plt
import numpy


class Utils():
	@staticmethod
	def bar_graph(name_value_dict, graph_title='', output_name='bargraph.png'):
	    figure(figsize=(8, 4)) # image dimensions   
	    title(graph_title, size='x-small')
	    
	    # add bars
	    for i, key in zip(range(len(name_value_dict)), name_value_dict.keys()):
	        bar(i + 0.25 , name_value_dict[key], color='red')
	    
	    # axis setup
	    xticks(arange(0.65, len(name_value_dict)), 
	        [('%s' % (name)) for name, value in 
	        zip(name_value_dict.keys(), name_value_dict.values())], 
	        size='xx-small')
	    max_value = 100#max(name_value_dict.values())
	    tick_range = arange(0, max_value, max_value/10)
	    yticks(tick_range, size='xx-small')
	    formatter = FixedFormatter([str(x) for x in tick_range])
	    gca().yaxis.set_major_formatter(formatter)
	    gca().yaxis.grid(which='major') 
	    
	    savefig(output_name)


class TestCorpus():
	
	# static variables common to all instances

	feature_words = stopwords.words('english')

	feature_types = {'BOOLEAN':0, 'FREQUENCY':1, 'FREQUENCY_NORMALIZED':2}
	# default feature type is FREQUENCY_NORMALIZED
	feature_type = 2

	classifier_types = {'NAIVE_BAYES':0, 'SVM_LINEAR':1, 'SVM_POLY':2}
	# default classifier is polynomial SVM
	classifier_type = 1


	# boolean values (feature word occurs or does not occur in text)
	@classmethod
	def features_boolean(cls, text, features=[]):
		if not features:
			features = cls.feature_words
		return dict((word, int(word in text)) for word in features)

	# frequency values, normalized (how many times a feature word occurs in text, normalized by text length)
	@classmethod
	def features_frequency_normalized(cls, text, features=[]):
		if not features:
			features = cls.feature_words
		# multiply normalized frequency count by 1000 to avoid very small numbers
		return dict((word, 1000.0*text.count(word)/float(len(text))) for word in features)

	# frequency values, not normalized
	@classmethod
	def features_frequency(cls, text, features=[]):
		if not features:
			features = cls.feature_words
		return dict((word, text.count(word)) for word in features)


	# takes as input lists of (text, label) pairs for each class (1/2), for training/testing
	def __init__(self, train_set_class1, train_set_class2, test_set_class1, test_set_class2):
		self.train_set_class1 = train_set_class1
		self.train_set_class2 = train_set_class2
		self.test_set_class1 = test_set_class1
		self.test_set_class2 = test_set_class2
		self.train_set = self.train_set_class1 + self.train_set_class2
		self.test_set = self.test_set_class1 + self.test_set_class2

		# use default feature type to compute list of features and labels for training set and test set
		self.train_feature_set = [(self.features_frequency_normalized(word_tokenize(text)), label) for (text,label) in self.train_set]
		self.test_feature_set = [(self.features_frequency_normalized(word_tokenize(text)), label) for (text,label) in self.test_set]	

		# custom feature_sets initialized with defaut
		self.train_feature_set_custom = self.train_feature_set
		self.test_feature_set_custom = self.test_feature_set
		self.feature_words_custom = self.feature_words

	# recompute featuresets with current parameters
	def compute_featuresets(self):
		if (self.feature_type == 0):
			self.train_feature_set_custom = [(self.features_boolean(word_tokenize(text), self.feature_words_custom), label) for (text,label) in self.train_set]
			self.test_feature_set_custom = [(self.features_boolean(word_tokenize(text), self.feature_words_custom), label) for (text,label) in self.test_set]	
		if (self.feature_type == 1):
			self.train_feature_set_custom = [(self.features_frequency(word_tokenize(text), self.feature_words_custom), label) for (text,label) in self.train_set]
			self.test_feature_set_custom = [(self.features_frequency(word_tokenize(text), self.feature_words_custom), label) for (text,label) in self.test_set]	
		if (self.feature_type == 2):
			self.train_feature_set_custom = [(self.features_frequency_normalized(word_tokenize(text), self.feature_words_custom), label) for (text,label) in self.train_set]
			self.test_feature_set_custom = [(self.features_frequency_normalized(word_tokenize(text), self.feature_words_custom), label) for (text,label) in self.test_set]

	# compute featuresets with current parameters separately for each (training) class
	def compute_class_featuresets(self):
		if (self.feature_type == 0):
			self.train_feature_set_class1_custom = [(self.features_boolean(word_tokenize(text), self.feature_words_custom), label) for (text,label) in self.train_set_class1]
			self.train_feature_set_class2_custom = [(self.features_boolean(word_tokenize(text), self.feature_words_custom), label) for (text,label) in self.train_set_class2]
		if (self.feature_type == 1):
			self.train_feature_set_class1_custom = [(self.features_frequency(word_tokenize(text), self.feature_words_custom), label) for (text,label) in self.train_set_class1]
			self.train_feature_set_class2_custom = [(self.features_frequency(word_tokenize(text), self.feature_words_custom), label) for (text,label) in self.train_set_class2]
		if (self.feature_type == 2):
			self.train_feature_set_class1_custom = [(self.features_frequency_normalized(word_tokenize(text), self.feature_words_custom), label) for (text,label) in self.train_set_class1]
			self.train_feature_set_class2_custom = [(self.features_frequency_normalized(word_tokenize(text), self.feature_words_custom), label) for (text,label) in self.train_set_class2]
	
	# mean frequency of feature words in texts
	def mean_features(self, featureset=[]):
		mean_features = {}
		if not featureset:
			featureset = self.train_feature_set + self.test_feature_set
		nr_ex = len(featureset)
		for stopword in featureset[0][0]:
			mean_features[stopword] = 0
			for train_example in featureset:
				mean_features[stopword] += train_example[0][stopword]/float(nr_ex)
				
		return mean_features

	# top feature words occuring in whole corpus
	def top_features(self, how_many=127):
		features = self.mean_features()
		# return same format dictionary only with top occuring features (according to mean_features result)
		return dict(sorted(features.iteritems(), key=operator.itemgetter(1), reverse = True)[:how_many])

	def set_nr_features(self, nr_features):
		top_swords = self.top_features(nr_features)
		self.feature_words_custom = top_swords.keys()
		# recompute featuresets for new feature vector
		self.compute_featuresets()
		print 'Nr of features: ', len(self.feature_words_custom)

	def set_feature_type(self, feature_type):
		self.feature_type = feature_type
		self.compute_featuresets()
		print 'Feature type: ', [name for name, value in self.feature_types.iteritems() if value==feature_type][0]

	def train_classifier(self, trainset=[], svm_param=1.0):
		# default train set is class field (all train files)
		if not trainset:
			trainset = self.train_feature_set_custom

		if (self.classifier_type == 0):
			self.classifier = SklearnClassifier(MultinomialNB())
			print "Training Naive Bayes classifier..."
		if (self.classifier_type == 1):
			self.classifier = SklearnClassifier(LinearSVC(penalty='l2', loss='l2', dual=False, C=svm_param, class_weight='auto'))
			print "Training Linear SVM classifier..."
		if (self.classifier_type == 2):
			self.classifier = SklearnClassifier(SVC(kernel='poly', C=svm_param, class_weight='auto'))
			print "Training Polynomial SVM classifier..."

		self.classifier.train(self.train_feature_set_custom)


	def testall_accuracy(self, testset=[]):
		# default test set is class field (all test files)
		if not testset:
			testset = self.test_feature_set_custom
		print 'Measuring classifier performance...'
		acc = accuracy(self.classifier, self.test_feature_set_custom)
		print 'Overall accuracy:', acc
		
		return acc

	def results_per_file(self, filenames=[]):
		# if no filenames are given as parameters just use numbers from 1 to nr_of_files
		if not filenames:
			filenames = range(len(self.test_feature_set_custom) + 1)[1:]
		print 'Results per file:'
		findex = 0
		# first index - element to be tested
		# second index - 0 = index of feature dictionary
		for text in self.test_feature_set_custom:
			predicted_label = self.classifier.classify(text[0])
			actual_label = text[1]
			print filenames[findex], predicted_label, predicted_label == actual_label
			findex += 1

	def classify_this(self, text):
		return self.classifier.classify(text)

	# TODO: easier computation of top_features and set_feature_nr. \
	# maybe just get first elements of sorted featuresets, not compute them again everytime

	def leave_one_out(self, feature_type=2, classifier_type=1, C=1.0, nr_features=127):
		print '\nCross-validating with leave-one-out...'

		# set parameters

		# if (nr_features != 127):
		# 	self.set_nr_features(nr_features)
		# if (feature_type !=2):
		# 	self.set_feature_type(feature_type)

		# faster: don't recompute featuresets everytime:

		self.feature_type = feature_type
		if (nr_features != 127):
			top_swords = self.top_features(nr_features)
			self.feature_words_custom = top_swords.keys()
		if (nr_features != 127 or feature_type != 2):
			self.compute_featuresets()
		print '\nNr features: ', nr_features
		print 'Feature type: ', \
		[name for name, value in self.feature_types.iteritems() if value==feature_type][0], '(%d)'%feature_type, '\n'
		self.classifier_type = classifier_type


		# cross-validate

		nrcorrect = 0
		total = len(self.train_feature_set_custom)
		for i in range (total):
			trainset = self.train_feature_set_custom[:i] + self.train_feature_set_custom[i+1:]
			self.train_classifier(trainset=trainset, svm_param=C)
			label = self.classify_this(self.train_feature_set_custom[i][0])
			print 'Testing on: file', i+1
			print 'actual: ', self.train_feature_set_custom[i][1]
			print 'predicted: ', label
			print "--------------------------------"
			if (label== self.train_feature_set_custom[i][1]):
				nrcorrect += 1
		print 'Correctly classified: ', nrcorrect, '/', total, '\n'
		return float(nrcorrect)/total

	# cross-validate results with leave-one-out for different parameters
	def cross_validate(self, validate_type=0):

		# validate_type =
		#				0: nr of features
		#				1: feature_type
		#				2: classifier_type
		#				3: classifier_parameter

		if (validate_type==0):
			# cross-validate for nr of features:
			# [stopwords, accuracies] = self.nrstopwords_experiment(True)
			# results = dict((stopwords[i], accuracies[i]) for i in range(len(stopwords)))
			nr_stopwords = range(1,100,10)
			results = dict((nr,0) for nr in nr_stopwords)
			for nr in nr_stopwords:
				acc = self.leave_one_out(nr_features=nr)
				results[nr] = acc

		# TODO: strange results for this? too accurate; weak methods too successful
		if (validate_type==1):
			# cross-validate for feature type
			results = dict((feat,0) for feat in self.feature_types)
			for feat in self.feature_types:
				acc = self.leave_one_out(feature_type=self.feature_types[feat])
				results[feat] = acc

		if (validate_type==2):
			# cross-validate for classifier type
			results = dict((cl, 0) for cl in self.classifier_types)
			for cl in self.classifier_types:
				acc = self.leave_one_out(classifier_type=self.classifier_types[cl])
				results[cl] = acc

		if (validate_type==3):
			# cross-validate for classifier parameter
			Cs = [10**(-10), 10**(-5), 10**(-3), 10**(-1), 1.0, 1.5, 10, 100, 1000, 10**5, 10**10]
			results = dict((C, 0) for C in Cs)
			for C in Cs:
				acc = self.leave_one_out(C=C)
				results[C] = acc

		return results


	# accuracy vs number of stopwords used
	def nrstopwords_experiment(self, validate=False):
		#TODO: not sure about these results,  maybe test some more
		accuracies = []
		for nr_stopwords in range(1,10):
			if (validate):
				acc = self.leave_one_out(nr_features=nr_stopwords)
			else:
				self.set_nr_features(nr_stopwords)
				self.train_classifier()
				acc = self.testall_accuracy()
			accuracies.append(acc)
			print nr_stopwords, acc

		for nr_stopwords in range(10,40,5):
			if (validate):
				acc = self.leave_one_out(nr_features=nr_stopwords)
			else:
				self.set_nr_features(nr_stopwords)
				self.train_classifier()
				acc = self.testall_accuracy()
			accuracies.append(acc)
			print nr_stopwords, acc

		for nr_stopwords in range(40,127,25):
			if (validate):
				acc = self.leave_one_out(nr_features=nr_stopwords)
			else:
				self.set_nr_features(nr_stopwords)
				self.train_classifier()
				acc = self.testall_accuracy()
			accuracies.append(acc)
			print nr_stopwords, acc

		stopwords = range(1,10) + range(10,40,5) + range(40,127,25)

		return [stopwords, accuracies]


	def plot_stopwords_vs_accuracy(self, validate=False):
		[stopwords, accuracies] = self.nrstopwords_experiment(validate)
		plt.plot(stopwords, accuracies, label='Circle')
		plt.xlabel('Nr of stopwords')
		plt.ylabel('Accuracy')
		plt.title('Performance of algorithm versus number of stopwords used in classification')
		plt.show()
		# save to disk
		#plt.savefig('stopwords_experiment2.png')

	def plot_featureword_distribution(self, nr_swords=25):
		self.compute_class_featuresets()
		# for test set and each class of train set
		Utils.bar_graph(self.mean_features(self.test_feature_set_custom), graph_title='%d stop words for test set - mean occurences'%nr_swords, output_name='test%d.png'%nr_swords)
		Utils.bar_graph(self.mean_features(self.train_feature_set_custom), graph_title='%d stop words for train set - mean occurences'%nr_swords, output_name='train%d.png'%nr_swords)
		Utils.bar_graph(self.mean_features(self.train_feature_set_class1_custom), graph_title='%(nr)d stop words for %(class)s set - mean occurences'%{'class':self.train_feature_set_class1_custom[0][1], 'nr':nr_swords}, output_name='hamilton%d.png'%nr_swords)
		Utils.bar_graph(self.mean_features(self.train_feature_set_class2_custom), graph_title='%(nr)d stop words for %(class)s set - mean occurences'%{'class':self.train_feature_set_class2_custom[0][1], 'nr':nr_swords}, output_name='madison%d.png'%nr_swords)


# class for Federalist papers corpus - static methods to get the files and labels them

class FederalistPapers():

	test_filenames = ['paper49', 'paper50', 'paper51', 'paper52', 'paper53', 'paper54', 'paper55', 'paper56', 'paper57', 'paper58', 'paper62', 'paper63']

	@staticmethod
	def get_trainset_class1():
		trainmadisonfile = open("fed_papers/fedpapers_train_madison.txt")
		trainmadison = re.split("======== .* ========", trainmadisonfile.read())
		trainmadisonfile.close()
		trainmadison = trainmadison[1:]
		return ([(text, 'madison') for text in trainmadison])

	@staticmethod
	def get_trainset_class2():
		trainhamiltonfile = open("fed_papers/fedpapers_train_hamilton.txt")
		trainhamilton = re.split("======== .* ========", trainhamiltonfile.read())
		trainhamiltonfile.close()
		trainhamilton = trainhamilton[1:]
		return ([(text, 'hamilton') for text in trainhamilton])

	@staticmethod
	def get_testset():
		testmadisonfile = open("fed_papers/fedpapers_test.txt")
		testmadison = re.split("======== .* ========", testmadisonfile.read())
		testmadisonfile.close()
		testmadison = testmadison[1:]

		test = ([(text, 'madison') for text in testmadison])
		return test



if __name__ == '__main__':
	# test for Federalist Papers

	train1 = FederalistPapers.get_trainset_class1()
	train2 = FederalistPapers.get_trainset_class2()
	test = FederalistPapers.get_testset()

	filenames = FederalistPapers.test_filenames

	corpus = TestCorpus(train1, train2, test, [])
	corpus.set_nr_features(25)
	corpus.classifier_type = TestCorpus.classifier_types['SVM_POLY']

	corpus.feature_type = TestCorpus.feature_types['FREQUENCY_NORMALIZED']
	corpus.compute_featuresets()
	#print 'features: ', corpus.feature_words_custom, len(corpus.feature_words_custom)

	corpus.train_classifier()
	corpus.testall_accuracy()
	corpus.results_per_file(filenames)

	#corpus.plot_stopwords_vs_accuracy(True)
	#pprint(corpus.cross_validate(3))
