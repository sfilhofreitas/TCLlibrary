"""
This module implements the Learner class

The Learner represents an entity with the following
interface (set of methods): start, fit, predict, get_params

The Learner is one of the three components (Teacher, Learner, Data)
of a comunication protocol. The protocol performs the interaction
between a Teacher and a Learner.

Author: Pedro Lazéra Cardoso
"""
import numpy as np

from .Definitions import get_qtd_rows
from .Definitions import InputSpace
from .Definitions import Labels
from .Definitions import join_input_spaces
from .Definitions import join_labels

class Learner:
	"""
	A class to represent a Learner, an entity with the following
	interface (set of methods): start, fit, predict, get_params

	Methods
	-----------
	start()
		Preprocess some data, if needed.
		Signals to the Learner that the teaching will start

	fit(X: InputSpace, y: Labels) -> None
		fit the data (matrix) X to labels (vector) y

	predict(X: InputSpace) -> Labels:
		apply the current model to the data X

	get_params() -> dict
		returns the parameters used by the Learner
	"""
	name = "GenericLearner"	

	def start(self):
		"""Just starts the Learner. Only useful it the learner
		has some kind of preprocessing"""
		pass

	def fit(self, X: InputSpace, y: Labels) -> None:
		""" Fit the data (matrix) X to labels (vector) y
		The model Learner model is updated
		Returns nothing (None)

		Parameters
		-----------
		X: InputSpace -- the data (features values), a matrix, where
						 each row is an example
		y: Labels -- the correct class for each example
		"""
		raise NotImplementedError

	def predict(self, X: InputSpace) -> Labels:
		""" Predicts, according to the learner model, the class
		of each example in X, the data

		Parameters
		-----------
		X: InputSpace -- the data (features values), a matrix, where
						 each row is an example
		"""
		raise NotImplementedError

	def get_params(self) -> dict:
		"""Returns the set of parameters in the learner configuration"""
		return dict()