
import numpy as np
from ..GenericTeacher import Teacher
from ..Utils.Sampler import get_first_examples
from sklearn import preprocessing
import warnings

class ALTCT(Teacher):
	"""docstring for FixedPercWrongTeacher"""
	name = "ALBasedTCT"
	_SEED = 0
	_FRAC_START = 0.01
	_ALPHA = 0.1
	_STATE_DONE = False

	_STATE_SEND_NEW_IDS = 0
	_STATE_CHOOSE_BATCH_SIZE_NEW_IDS = 1
	_STATE_SEND_EMPTY_NEW_IDS = 2
	
	def __init__(self, seed: int = _SEED,
		frac_start: float = _FRAC_START,
		alpha = _ALPHA):
		self.seed = seed
		self.frac_start = frac_start		
		self.alpha = alpha
		
		

	def start(self, X, y, time_left: float):
		self._start(X, y, time_left)
		self.num_iters = 0
		self.m = y.size
		self.S_current_size = 0		
		self.state_new_ids = self._STATE_SEND_NEW_IDS
		self._STATE_DONE = False
		self.f_shuffle = np.random.RandomState(self.seed).shuffle
		self.shuffled_ids = self._get_shuffled_ids()
		self.last_accuracy = 0.0
		self.best_accuracy = -np.inf

		assert len(self.shuffled_ids) == len(self.ids)

	def _get_shuffled_ids(self):
		ids = np.arange(self.m, dtype=int)
		# f_shuffle = np.random.RandomState(self.seed).shuffle
		self.f_shuffle(ids)
		return ids

	def _keep_going(self):
		return self.S_current_size < self.m

	def get_first_examples(self, time_left: float):
		classes = np.unique(self.y) # isso devia sair. devia ser computado for get_first_examples
		f_shuffle = np.random.RandomState(self.seed).shuffle
		new_ids = get_first_examples(self.frac_start, self.m, classes, self.y, f_shuffle)
		new_ids = np.array(new_ids)
		
		# update shuffled_ids. Aqui usamos ponteiros de ponteiros.
		# debugue as próximas duas linhas de cabeça vazia
		_new_ids = set(new_ids)
		self.shuffled_ids = np.append(new_ids,
							          [i for i in self.shuffled_ids if i not in _new_ids])
		
		# update batch size, from 1 to len(new_ids), based on strategy		
		self.batch_size = len(new_ids)
		

		return self._send_new_ids(new_ids)

	
	def get_new_examples(self, test_ids, test_labels, test_probabilities, time_left: float):
		if not self._keep_going() or self._STATE_DONE:
			return np.array([])

		sorted_probabilities = [np.sort(x) for x in test_probabilities]
		smallest_margin_list = [x[-1]-x[-2] for x in sorted_probabilities]
		
		k = min(int(self.batch_size*self.alpha), self.m-self.batch_size-1)
		sorted_ids = test_ids[np.argpartition(smallest_margin_list, k)]		
		self.f_shuffle(sorted_ids[k:])
		new_ids = sorted_ids[:self.batch_size]

		unselected_ids = np.copy(sorted_ids[self.batch_size:]) 		
		remaining_ids = np.copy(self.shuffled_ids[self.S_current_size+test_ids.size:])
		self.shuffled_ids[self.S_current_size:] = np.concatenate((new_ids, remaining_ids, unselected_ids))

		correct_test_labels = self.y[test_ids]
		wrong_ids = test_ids[test_labels != correct_test_labels]

		error = (len(wrong_ids)/len(test_ids))
		self.last_accuracy = 1.0 - error
		self.last_accuracy -= 1.96*np.sqrt(self.last_accuracy*(1-self.last_accuracy)/len(test_ids))
		

		self.batch_size *= 2
		self.state_new_ids = self._STATE_SEND_NEW_IDS

		return self._send_new_ids(new_ids)


	def get_new_test_ids(self, test_ids,
		test_labels, time_left: float) -> np.ndarray:
		if not self._keep_going():
			return np.array([])

		if self.state_new_ids == self._STATE_SEND_NEW_IDS:
			self.sample_size = 2*self.S_current_size
			_start = self.S_current_size 
			_end = min(_start + self.sample_size, self.m)
			new_ids = self.shuffled_ids[_start:_end]
			self.state_new_ids = self._STATE_SEND_EMPTY_NEW_IDS
			assert type(new_ids) == type(np.array([]))
			return new_ids
		
		return np.array([])

	def get_log_header(self):
		return ["iter_number", "training_set_size", "accuracy"]

	def get_log_line(self, h):
		accuracy = 1 - self._get_wrong_labels_id(h).size/self.y.size
		log_line = [self.num_iters, self.S_current_size, accuracy]
		return log_line

	def _send_new_ids(self, new_ids):
		self.num_iters += 1
		self.S_current_size += len(new_ids)
		#self.f_shuffle(new_ids)
		return new_ids


	def get_params(self) -> dict:
		return {
			"seed": self.seed,
			"frac_start": self.frac_start,			
			"alpha": self.alpha,			
			}

	def get_estimated_accuracy(self, test_ids=None, test_labels=None):
		if len(test_ids) == 0:
			return 0			
		return self.last_accuracy

		
	def is_best_model(self, test_ids=None,
		test_labels=None):
		if self.last_accuracy > (self.best_accuracy+0.000000001):
			self.best_accuracy = self.last_accuracy
			return True		
		return False



