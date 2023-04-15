"""
This module implements the function protocol.

The function performs the interactions between
a teacher T and a learner L over a dataset (X, X_labels),
where X is a vector of rows of atributs and X_labels
is a vector of labels, one for each row in X

In the first interaction, T provides
some examples (X_0, y_0) that are given to L. L trains with
(fits) this first set of examples

Every other interaction has two parts. In the first parts,
T provides examples X_t (subset of X) and L classifies (does not 
train with) these examples. In the second part, aware of
how L classified each example in X_t, T provides
some examples (X_s, y_s) that are given to L. L trains with
(fits) this set of examples.

The total time spent in the interactions must not exceed a time
limit.

Author: Pedro Lazéra Cardoso
"""

import numpy as np
from datetime import datetime
from copy import copy
from sklearn.utils import shuffle
from .Utils.TeachResult import TeachResult
from .Utils.Timer import Timer

from .GenericTeacher import Teacher
from .GenericLearner import Learner

from .Definitions import InputSpace
from .Definitions import Labels
from .Definitions import wrapp_labels
from .Definitions import wrapp_input_space
from .Definitions import get_qtd_columns
from .Definitions import get_qtd_rows

from copy import deepcopy 
from func_timeout import func_timeout, FunctionTimedOut

from sklearn.exceptions import NotFittedError

_TIMER_KEYS = ("training", "classification", "get_examples")



_LOG_HEADER = ["iter", "TS_size", "dataset_accuracy", "elapsed_time",
	"time_left", "get_examples_time", "training_time",
	"classification_time", "qtd_classified_examples", 
	"test_set_accuracy", "estimated_accuracy", 
	"learner_selected", "accuracy_selected"]


_IND_TEST_ACC = _LOG_HEADER.index('test_set_accuracy')



_TIME_LIMIT = 1000000000.0 # in seconds

_SHUFFLE_RANDOM_STATE = 0
_SHUFFLE_DATASET = False


def teach(T: Teacher, L: Learner,
	X: InputSpace, X_labels: Labels, 
	X_test: InputSpace = None, X_test_labels: Labels = None,
	time_limit = _TIME_LIMIT,	
	send_probabilities: bool = False) -> TeachResult:
	# timer
	timer = Timer()	
	timer.start()
	get_time_left = lambda: time_limit - timer.get_elapsed_time()
	_set_timer_keys_to_zero(timer, _TIMER_KEYS)

	# teacher log	
	test_ids = np.array([], dtype=int)

	result = TeachResult()
	result.__add_log__(_LOG_HEADER)

	# wrappers
	X = wrapp_input_space(X)
	X_labels = wrapp_labels(X_labels)

	# checks
	global classes 
	classes = np.unique(X_labels)
	assert len(classes) > 1, 'there must be more than one label in the label vector'
	assert np.min(classes) == 0, 'Labels must be in the range [0, |Y|-1]'
	assert np.max(classes) < len(classes), 'Labels must be in the range [0, |Y|-1]'
	result.classes = np.array(classes[:])

	# start with empty set of <training example ids>
	train_ids = np.array([], dtype=int)
	
	# initialization	
	T.start(X, X_labels, get_time_left())

	# first teaching interaction

	## get first examples
	timer.tick("get_examples")
	new_train_ids = T.get_first_examples(get_time_left())
	assert 0 < len(new_train_ids) <=  get_qtd_rows(X)
	timer.tock()

	iter_selected_learner = 1
		
	# other teaching interactions
	qtd_iters = 0
	while (get_time_left() > 0):
		
		qtd_iters += 1

		if len(new_train_ids) > 0:
			
			timer.tick("training")
			
			train_ids = np.append(train_ids, new_train_ids)			
			
			assert len(train_ids) <= get_qtd_rows(X)
			
			try:
				func_timeout(get_time_left(), L.fit, args=(X[train_ids],X_labels[train_ids]))
				result.__add_model__(deepcopy(L))
				result.__add_selected_examples__(new_train_ids)
			except FunctionTimedOut:    					
				break					
			
			timer.tock()
			
		else:
			break

		
		# run next iteration
		timer.tick("classification")
		test_ids, test_labels, *extra_infos = _run_tests(T, L, X, send_probabilities, get_time_left)
		timer.tock()
		
		timer.tick("get_examples")
		new_train_ids = T.get_new_examples(test_ids, test_labels, *extra_infos, get_time_left())
		timer.tock()

		
		current_accuracy = T.get_estimated_accuracy(test_ids, test_labels)

		#futuramente colocar um conjunto de validacao
		if T.is_best_model(test_ids, test_labels): 
			iter_selected_learner = qtd_iters
			result.__set_model__(deepcopy(L))
			result.__set_model_id__(qtd_iters-1)		
		
		timer.stop()	
		_log_line = _get_log_line(L, X, X_labels, X_test, X_test_labels, 
		 	train_ids, test_ids, timer, get_time_left(), qtd_iters)

		#a linha do log do learner atual ainda vai ser adicionada a seguir			
		if  iter_selected_learner == len(result.log):
			selected_accuracy = _log_line[_IND_TEST_ACC]
		else:
			selected_accuracy = result.log[iter_selected_learner][_IND_TEST_ACC]
		
		_log_line = _log_line + [current_accuracy, iter_selected_learner-1, selected_accuracy]			
		result.__add_log__(_log_line)
		timer.unstop()

		


		
	# sanity checks
	assert len(train_ids) == len(set(train_ids))
	if qtd_iters < 1:		
		raise NotFittedError(f'This %s instance is not fitted yet.' % (type(L).__name__))

	return result


def _run_tests(T: Teacher, L: Learner,
	X: InputSpace, send_probabilities, get_time_left):
	test_ids = np.array([], dtype=int)
	test_labels = np.array([], dtype=int)
	test_probabilities = np.empty(shape=(0,len(classes)))

	while len(test_ids) <= get_qtd_rows(X):
		new_test_ids = T.get_new_test_ids(test_ids, test_labels, get_time_left())
		if len(new_test_ids) > 0:
			assert len(new_test_ids) + len(test_ids) <= get_qtd_rows(X)
			
			if send_probabilities:
				probabilities = L.predict_proba(X[new_test_ids])
				new_test_labels = [np.argmax(x) for x in probabilities]
				test_probabilities = np.append(test_probabilities, probabilities, axis=0)
			else:
				new_test_labels = L.predict(X[new_test_ids])

			test_ids = np.append(test_ids, new_test_ids)
			test_labels = np.append(test_labels, new_test_labels)
		else:
			break

	if send_probabilities:
		return (test_ids, test_labels, test_probabilities)

	return (test_ids, test_labels)


def _get_log_line(L: Learner,
	X: InputSpace, X_labels: Labels, 
	X_test: InputSpace, X_test_labels: Labels,
	train_ids, test_ids, timer, time_left, qtd_iters):
	accuracy = _get_accuracy(L.predict(X), X_labels)
	#qtd_classes, dist_classes = _get_class_qtd_and_distribution(X_labels[train_ids])
	
	if X_test is not None:
		test_set_accuracy = _get_accuracy(L.predict(X_test), X_test_labels)
	else:
		test_set_accuracy = '-'

	
	log_line = [
		qtd_iters,
		len(train_ids),
		accuracy,
		timer.get_elapsed_time(),
		time_left,
		timer["get_examples"],
		timer["training"],
		timer["classification"],
		len(test_ids),
		#qtd_classes,
		#dist_classes,
		test_set_accuracy
	]

	return log_line

# def _get_class_qtd_and_distribution(labels):
# 	qtd_classes = len(np.unique(labels))
# 	dist_classes = np.bincount(labels) / len(labels)
# 	dist_classes = ",".join("{:.2f}".format(i) for i in dist_classes)
# 	return (qtd_classes, dist_classes)

def _set_timer_keys_to_zero(timer, keys):
	for key in keys:
		timer.tick(key)
		timer.tock()

def _get_accuracy(y, h):
	assert len(y) == len(h)
	qtd_wrong_labels = np.count_nonzero(y != h)
	accuracy = 1 - qtd_wrong_labels / len(y)
	return accuracy



# _LOG_HEADER = ["iter", "TS_size", "dataset_accuracy", "elapsed_time",
# 	"time_left", "get_examples_time", "training_time",
# 	"classification_time", "qtd_classified_examples", "TS_qtd_classes",
# 	"TS_class_distribution", "test_set_accuracy", "estimated_accuracy", 
# 	"estimate_set_size", "learner_selected", "accuracy_selected"]



# def teach(T: Teacher, L: Learner,
# 	X: InputSpace, X_labels: Labels, 
# 	X_test: InputSpace = None, X_test_labels: Labels = None,
# 	time_limit = _TIME_LIMIT,	
# 	save_best_learner: bool = False,
# 	send_probabilities: bool = False) -> TeachResult:
# 	# timer
# 	timer = Timer()
# 	ok_timer = None
# 	timer.start()
# 	get_time_left = lambda: time_limit - timer.get_elapsed_time()
# 	_set_timer_keys_to_zero(timer, _TIMER_KEYS)

# 	# teacher log	
# 	test_ids = np.array([], dtype=int)

# 	result = TeachResult()
# 	result.__add_log__(_LOG_HEADER)

# 	# wrappers
# 	X = wrapp_input_space(X)
# 	X_labels = wrapp_labels(X_labels)

# 	# checks
# 	global classes 
# 	classes = np.unique(X_labels)
# 	assert len(classes) > 1, 'there must be more than one label in the label vector'
# 	assert np.min(classes) == 0, 'Labels must be in the range [0, |Y|-1]'
# 	assert np.max(classes) < len(classes), 'Labels must be in the range [0, |Y|-1]'

# 	# start with empty set of <training example ids>
# 	train_ids = np.array([], dtype=int)
# 	ok_train_ids = None

# 	# initialization	
# 	T.start(X, X_labels, get_time_left())

# 	# first teaching interaction

# 	## get first examples
# 	timer.tick("get_examples")
# 	new_train_ids = T.get_first_examples(get_time_left())
# 	assert 0 < len(new_train_ids) <=  get_qtd_rows(X)
# 	timer.tock()

# 	result.__add_selected_examples__(new_train_ids)


# 	## fit first examples
# 	train_ids = np.append(train_ids, new_train_ids)
		
# 	timer.tick("training")
# 	try:
		
# 		func_timeout(get_time_left(), L.fit, args=(X[train_ids],X_labels[train_ids]))
# 		result.__add_model__(deepcopy(L))
		
# 	except FunctionTimedOut:
# 		pass
		
# 	timer.tock()
# 	best_accuracy = 0
# 	iter_selected_learner = 1
	
	
# 	# other teaching interactions
# 	qtd_iters = 0
# 	while (get_time_left() > 0):
# 		# copy last "ok" state and build log line
# 		qtd_iters += 1
# 		timer.stop()
# 		ok_timer = copy(timer)
# 		ok_timer.finish()
# 		ok_train_ids = train_ids[:]
# 		_log_line = _get_log_line(L, X, X_labels, X_test, X_test_labels, 
# 			ok_train_ids, test_ids, ok_timer, get_time_left(), qtd_iters)
# 		if not save_best_learner:
# 			iter_selected_learner = qtd_iters			
# 			result.__add_log__(_log_line+[0,0,qtd_iters, _log_line[_IND_TEST_ACC]])
# 		timer.unstop()

# 		# run next iteration
# 		timer.tick("classification")
# 		test_ids, test_labels, *extra_infos = _run_tests(T, L, X, send_probabilities, get_time_left)
# 		timer.tock()
		
# 		timer.tick("get_examples")
# 		new_train_ids = T.get_new_examples(test_ids, test_labels, *extra_infos, get_time_left())
# 		timer.tock()


# 		#futuramente colocar um conjunto de validacao
# 		if save_best_learner: 
# 			#caso em que todos os exemplos ja foram explorados
# 			#forcar retornar o ultimo learner
# 			if len(new_train_ids) == 0:
# 				current_accuracy = 0
# 			else:	
# 				current_accuracy = T.get_estimated_accuracy(test_ids, test_labels)
# 				#current_accuracy -= 1.96*np.sqrt(current_accuracy*(1-current_accuracy)/len(test_ids))
			

# 			if (current_accuracy + 0.00001 >= best_accuracy):
# 				best_accuracy = current_accuracy				
# 				iter_selected_learner = qtd_iters
# 				result.__set_model__(deepcopy(L))
# 				result.__set_model_id__(qtd_iters-1)

# 			#a linha do log do learner atual ainda vai ser adicionada a seguir			
# 			if  iter_selected_learner == len(result.log):
# 				selected_accuracy = _log_line[_IND_TEST_ACC]				
# 			else:
# 				selected_accuracy = result.log[iter_selected_learner][_IND_TEST_ACC]
			
# 			_log_line = _log_line + [current_accuracy,len(test_ids), iter_selected_learner-1, selected_accuracy]			
# 			result.__add_log__(_log_line)
# 		else:			
# 			result.__set_model__(deepcopy(L))
# 			result.__set_model_id__(qtd_iters-1)


# 		if len(new_train_ids) > 0:
			
# 			timer.tick("training")
			
# 			train_ids = np.append(train_ids, new_train_ids)
# 			result.__add_selected_examples__(new_train_ids)
			
# 			assert len(train_ids) <= get_qtd_rows(X)

			
# 			try:
# 				func_timeout(get_time_left(), L.fit, args=(X[train_ids],X_labels[train_ids]))
# 				result.__add_model__(deepcopy(L))
# 			except FunctionTimedOut:    					
# 				break			
			
			
# 			timer.tock()
			
# 		else:
# 			break


		
# 	# sanity checks
# 	# assert qtd_iters >= 1, "there was no training..."
# 	# assert ok_timer is not None
# 	# assert ok_train_ids is not None
# 	#assert len(ok_train_ids) == len(set(ok_train_ids))
# 	if qtd_iters < 1:		
# 		raise NotFittedError(f'This %s instance is not fitted yet.' % (type(L).__name__))

# 	return result
