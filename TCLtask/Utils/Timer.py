"""
This module implements a stopwatch used in the Protocol module,
to measure the time spent by the Teacher and the Learner
in each process of the learning process

Author: Pedro Lazéra Cardoso
"""

from timeit import default_timer
from copy import deepcopy

class Timer:
	"""
	A class to represent a stopwatch with steroids

	Methods
	-----------
	start()
		Starts the stopwatch

	tick(field: str) -> None
		Begins (or continue) to count time spent in
		process "field". "field" is now the active process

	tock() -> Labels:
		Stops to count time spent in the active process

	finish() -> dict
		Stops to count time. Freezes the stopwatch.

	get_elapsed_time()
		Returns total elapsed time

	stop()
		Stops to count time

	unstop()
		Continues to count time
	"""
	_OFF_STATE = 0
	_ON_STATE = 1
	_TICK_STATE = 2
	_STOP_STATE = 3
	_FINISHED_STATE = 4

	def __init__(self):
		self._d = dict()
		self._state = Timer._OFF_STATE

	def start(self):
		""" Starts the clock. The special field "total_time"
		starts ticking. The dictionary of fields is erased.
		"""
		self.total_time = 0.0
		self.others_time = 0.0

		self._t0_total_time = default_timer()

		self._d.clear()

		self._state = Timer._ON_STATE

		self._curr_field = None
		self._t0_stop_time = None
		self._t0_curr_field = None

	def tick(self, field: str):
		""" Starts (or restarts) counting time of 'field' """
		if self._state == Timer._STOP_STATE:
			self.unstop()
		
		assert (self._state == Timer._ON_STATE), "cannot tick twice or tick before start"
		
		self._t0_curr_field = default_timer()
		self._curr_field = field

		self._state = Timer._TICK_STATE

	def tock(self):
		""" Stops counting time of the last ticked field """
		assert (self._state == Timer._TICK_STATE), "cannot tock before tick"

		curr_field = self._curr_field
		delta = default_timer() - self._t0_curr_field
		self._d[curr_field] = self._d.get(curr_field, 0.0) + delta

		self._state = Timer._ON_STATE
		self._curr_field = None

	def finish(self):
		"""  """
		assert (self._state in (Timer._ON_STATE,
			Timer._TICK_STATE, Timer._STOP_STATE)), "cannot finish before start"
		
		if self._state == Timer._TICK_STATE:
			self.tock()
		elif self._state == Timer._STOP_STATE:
			self.unstop()

		self.total_time = default_timer() - self._t0_total_time
		self.others_time = self.total_time - sum(self._d.values())

		self._state = Timer._FINISHED_STATE

	def get_elapsed_time(self) -> float:
		if self._state == Timer._OFF_STATE:
			elapsed_time = 0.0
		if self._state == Timer._FINISHED_STATE:
			elapsed_time = self.total_time
		elif self._state == Timer._STOP_STATE:
			elapsed_time = self._t0_stop_time - self._t0_total_time
		else:
			elapsed_time = default_timer() - self._t0_total_time
		
		return elapsed_time

	def stop(self):
		assert self._state in (Timer._ON_STATE, Timer._TICK_STATE)
		
		if self._state == Timer._TICK_STATE:
			self.tock()

		self._state = Timer._STOP_STATE
		self._t0_stop_time = default_timer()

	def unstop(self):
		assert self._state == Timer._STOP_STATE, "cannot unstop before stop"

		delta = default_timer() - self._t0_stop_time
		self._t0_total_time += delta

		self._state = Timer._ON_STATE

	def __str__(self):
		d_names = list(self._d.keys())
		d_values = [self._d[name] for name in d_names]
		v_names = ["total_time"] + d_names + ["others_time"]
		v_values = [self.total_time] + d_values + [self.others_time]
		s = '\n'.join('{} = {:.3f}'.format(n,v) for (n,v) in zip(v_names, v_values))
		return s

	def __add__(self, other):
		keys1 = list(self._d.keys())
		keys2 = list(other._d.keys())
		assert keys1 == keys2

		_d = dict()
		for k in keys1: # or keys2, whatever
			_d[k] = self._d[k] + other._d[k]

		total_time = self.total_time + other.total_time
		others_time = self.others_time + other.others_time

		new_timer = Timer()
		new_timer._d = _d
		new_timer.total_time = total_time
		new_timer.others_time = others_time

		return new_timer

	def __mul__(self, alpha):
		new_timer = Timer()

		for k in self._d.keys():
			new_timer._d[k] = self._d[k] * alpha

		new_timer.total_time = self.total_time * alpha
		new_timer.others_time = self.others_time * alpha

		return new_timer

	def __truediv__(self, alpha):
		return self.__mul__(1/alpha)

	def __getitem__(self, key):
		return self._d[key]

	def __copy__(self):
		other = Timer()
		other._d = deepcopy(self._d)

		other.total_time = self.total_time
		other.others_time = self.others_time
		other._t0_total_time = self._t0_total_time
		other._state = self._state
		other._curr_field = self._curr_field
		other._t0_stop_time = self._t0_stop_time
		other._t0_curr_field = self._t0_curr_field
		other._state = self._state

		return other
