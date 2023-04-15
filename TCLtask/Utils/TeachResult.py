"""
This modules implements the class TeachResult,
the result of a set of interactions between a teacher T
and a Learner L over a Dataset (X, y)

The result contains the teaching set ids (ids of the
examples) provided by T to L, so that L tries to fit
these examples, the hypothesis h (how L classifies
each example in the entire dataset) and some statistics
of the intereactions

Author: Pedro Lazéra Cardoso
"""

from datetime import datetime
from copy import deepcopy
from copy import copy
from math import isclose

from ..GenericTeacher import Teacher
from ..GenericLearner import Learner
from ..Definitions import Labels
from .Timer import Timer
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

class TeachResult:
	def __init__(self):
		self.models = []
		self.model = None
		self.model_id = None
		self.log = []
		self.selected_examples = []
		self.classes = []

	def __add_model__(self, model):
		self.models.append(model)

	
	def __set_model__(self, model):
		self.model = model

	
	def __set_model_id__(self, model_id):
		self.model_id = model_id

	
	def __add_log__(self, log):
		self.log.append(log)

	
	def __add_selected_examples__(self, new_examples):
		self.selected_examples.append(new_examples)

	
	def get_log_dataframe(self):
		if self.log == []:
			msg = "No log have been added to this instance."			
			raise NotFittedError(msg)

		df = pd.DataFrame(data=self.log[1:], columns=self.log[0])
		df.insert(0, 'model_id', np.arange(df.shape[0]))
		return df

	
	def get_teaching_set(self, model_id = None):
		if model_id is None:
			model_id = self.model_id

		if model_id is None:
			raise ValueError("model_id is None!")

		if (model_id < 0) or (model_id >= len(self.selected_examples)):
			raise IndexError(f'list index out of range: %s (size %s).' \
				             %(model_id, len(self.selected_examples)))
		
		return np.concatenate(self.selected_examples[:model_id+1])

	


	def predict_by_committee(self, X): 
		if len(self.models) == 0:
			msg = "No classifiers have been added to this instance. "			
			raise NotFittedError(msg)
		
		n_examples = X.shape[0]
		#ind_n_classes = self.log[0].index('TS_qtd_classes')
		#n_classes = self.log[-1][ind_n_classes]	    
		n_classes = len(self.classes)

		votes = np.zeros(shape=(n_examples, n_classes))
		for i, L in enumerate(self.models):        
			# i+1: the first line of the log is the header
			model_weight_index = self.log[0].index('estimated_accuracy')
			model_weight = self.log[i+1][model_weight_index]        
			model_weight = 1 if np.isclose(model_weight, 0) else model_weight

			pred_y = L.predict(X)
			votes[np.arange(n_examples),pred_y] += model_weight

		return np.argmax(votes, axis=1)



