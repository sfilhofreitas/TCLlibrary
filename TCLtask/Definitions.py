"""
This modules contains the basic types (or classes) used
in the protocol definied in the Protocol module. Along
with these type, some functions to manipulate and
extract statistics from objects of these types are provided

InputSpace -- a two dimensional array from numpy lib
Labels -- a one dimensional array from numpy lib

Author: Pedro Lazéra Cardoso
"""

import numpy as np

# type annotation
InputSpace = np.ndarray
Labels = np.ndarray

# numpy axis ids
_ROW_AXIS = 0
_COL_AXIS = 1

def get_qtd_rows(v) -> int:
	"""Returns the qtd of rows of the array v"""
	return v.shape[_ROW_AXIS]

def get_qtd_columns(v) -> int:
	"""Returns the qtd of columns of the array v"""
	return v.shape[_COL_AXIS]

# Input Space funtions

def join_input_spaces(X1: InputSpace, X2: InputSpace):
	"""Merges (concatenate) two input spaces. Is the same
	as stacking two matrices"""
	return np.vstack((X1, X2))

def wrapp_input_space(X: InputSpace):
	"""Transform an interable X in an InputSpace
	(two dimensional array from numpy lib)"""
	return np.array(X)

# Labels functions

def join_labels(y1: Labels, y2: Labels):
	"""Merges labels (vectors) y1 and y2. Is the same
	as stacking two vectors"""
	if y1.size == 0:
		return y2
	elif y2.size == 0:
		return y1

	return np.concatenate((y1, y2), axis=_ROW_AXIS)

def wrapp_labels(y: Labels):
	"""Transforms an interable y in a one dimensional
	array from numpy lib"""
	return np.array(y).reshape(-1)
