"""
This module implements the Teacher class

The Teacher represents an entity with the following
interface (set of methods):
- start
- get_first_examples
- get_new_examples
- get_new_test_ids
- get_params

The Teacher is one of the three components (Teacher, Learner, Data)
of a comunication protocol. The protocol performs the interaction
between a Teacher and a Learner.

Author: Pedro Lazéra Cardoso
"""

import numpy as np

from .Definitions import get_qtd_rows
from .Definitions import InputSpace
from .Definitions import Labels

class Teacher:
	"""
	A class to represent a Teacher, an entity with the following
	interface (set of methods):
		- start
		- get_first_examples
		- get_new_examples
		- get_new_test_ids
		- get_params

	Methods
	-----------
	start(X: InputSpace, y: Labels, time_left: float)
		Preprocess some data, if needed.
		Informs the teacher the set of examples (along with
		the correct labels for each example) it has to
		teach to a Learner

	get_first_examples(time_left: float)
		Returns the (ids of the) first examples that should
		be given to the learner

	get_new_examples(X: InputSpace) -> Labels:
		Returns a new set of (ids of) examples that should
		be given to the learner

	get_new_test_ids(test_ids,
		test_labels: Labels, time_left: float)
		Returns a new set of examples to test the Learner

	get_params() -> dict
		Returns the parameters used by the Teacher
	"""

	name = "GenericTeacher"

	def start(self, X: InputSpace, y: Labels, time_left: float):
		"""Starts the Teacher.
		Informs two things to the teacher:
		(1) the set of examples (along with the correct labels 
		for each example) it has to teach to a Learner

		The method must be implemented by the subclass

		Parameters
		-----------
		X: InputSpace -- the data (features values), a matrix, where
						 each row is an example
		y: Labels -- the correct class for each example
		time_left: float -- how much time is left for the interaction
		between the teacher and the learner
		"""
		raise NotImplementedError

	def get_first_examples(self, time_left: float) -> np.ndarray:
		"""
		Returns the (ids of the) first examples that should
		be given to the learner, so that the learner train
		with (fit) these examples

		This set of examples is especial, because it must be provided
		by the teacher before any feedback from the Learner

		The method must be implemented by the subclass

		Parameters
		-----------
		time_left: float -- how much time is left for the interaction
		between the teacher and the learner
		"""
		raise NotImplementedError

	def get_new_examples(self, test_ids,
		test_labels: Labels, time_left: float) -> np.ndarray:
		"""
		Returns a new set of (ids of) examples that should
		be given to the learner, so that the learner train
		with (fit) these examples

		This set of examples can be provided after some feedback
		from the learner. The feedback is how the learner classifies
		a set of examples (to classify, not to learn) choosen by
		the Teacher

		The method must be implemented by the subclass

		Parameters
		-----------
		X: InputSpace -- the data (features values), a matrix, where
						 each row is an example
		y: Labels -- the correct class for each example
		time_left: float -- how much time is left for the interaction
		between the teacher and the learner
		"""
		raise NotImplementedError

	def get_new_test_ids(self, test_ids,
		test_labels: Labels, time_left: float) -> np.ndarray:
		"""
		Returns a set of examples that should be given to
		the learner, so that the learner classify (predict the
		label of) these examples

		Parameters
		-----------
		test_ids -- vector of indexes, correspondind to examples ids
		in the dataset self.X
		test_labels: Labels -- the label (provided by the learner, not
		the correct label) for each example in test_ids
		time_left: float -- how much time is left for the interaction
		between the teacher and the learner
		"""
		if len(test_ids) == 0:
			return self.ids
		else:
			return []

	def get_params(self) -> dict:
		"""Returns the set of parameters in the teacher configuration"""
		return dict()

	def _start(self, X: InputSpace, y: Labels, time_left: float):
		"""The standard implementation of the start function,
		in case the subclass does not implement one.

		Just saves the features matrix X and the labels y of
		each row of X

		Parameters
		-----------
		X: InputSpace -- the data (features values), a matrix, where
						 each row is an example
		y: Labels -- the correct class for each example
		time_left: float -- how much time is left for the interaction
		between the teacher and the learner"""
		self.X = X
		self.y = y
		self.ids = np.arange(y.size, dtype=int)

		qtd_rows_X = get_qtd_rows(X)
		qtd_rows_y = get_qtd_rows(y)
		assert qtd_rows_X == qtd_rows_y

	def _get_wrong_labels_id(self, h: Labels):
		"""Returns the ids (row number) of the missclassified
		examples in the vector of labels h

		h should contain the labels of the entire dataset,
		which means that h has the same number of rows as the
		dataset X

		Parameters
		-----------
		h: Labels -> a vector of labels, with the same number of
		rows as the vector of correct labels self.y
		"""
		wrong_labels = self.y != h
		wrong_labels = wrong_labels.reshape(-1)
		return self.ids[wrong_labels]

	def _get_accuracy(self, h):
		"""Returns the accuracy of the classification h

		Parameters
		-----------
		h: Labels -> a vector of labels, with the same number of
		rows as the vector of correct labels self.y"""
		wrong_labels = self._get_wrong_labels_id(h)
		accuracy = 1 - len(wrong_labels) / len(self.y)
		return accuracy

	def get_estimated_accuracy(self, test_ids,
		test_labels):
		
		if len(test_ids) == 0:
			return 0 

		correct_test_labels = self.y[test_ids]
		wrong_ids = test_ids[test_labels != correct_test_labels]
		error = (len(wrong_ids)/len(test_ids))		
		
		return 1.0 - error

	def is_best_model(self, test_ids,
		test_labels):		
		return True		
		