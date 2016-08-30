'''Perceptron.py'''

#### Libraries ####
# Third Party Libraries 
import numpy as np

__version__ = "1"
__author__ = "Lukasz Obara"

class Perceptron(object):
	def __init__(self, data, threshold=0.5, learning_rate=0.1, epochs=50):
		self.data = data
		self.weights = np.zeros(len(data[0][0]))
		self.threshold = threshold
		self.learning_rate = learning_rate
		self.epochs = epochs

		def train(self):
		j = 0

		while True:
			error_count = 0

			for input_vector, result_vector in self.data:
				result = np.dot(input_vector, self.weights) > self.threshold
				error = result_vector - result

				if error != 0:
					error_count += 1
					self.weights += input_vector * error * self.learning_rate

			if error_count == 0 or j == self.epochs:
				if j == self.epochs:
					print('Limit reached: j = {}'.format(j))
				else:
					print('Error = 0, in {} steps'.format(j))
				break

			j += 1

		return self.weights

if __name__ == '__main__':
	# The data is structured similarly to the MNIST data set, noting 
	# that the data is arranged as a list containg a 2-tuple of arrays,
	# that is [(np.array(), np.array()), (np.array(), np.array()), ...]

	or_gate = [(np.array([0, 0, 1]), np.array([0])), 
			   (np.array([0, 1, 1]), np.array([1])), 
			   (np.array([1, 0, 1]), np.array([1])),
			   (np.array([1, 1, 1]), np.array([1]))]

	test = Perceptron(or_gate)
	print(test.train())

	# Example created to show that the xor_gate will not converge
	xor_gate = [(np.array([0, 0, 1]), np.array([0])), 
				(np.array([0, 1, 1]), np.array([1])), 
				(np.array([1, 0, 1]), np.array([1])), 
				(np.array([1, 1, 1]), np.array([0]))] 
	test = Perceptron(xor_gate)
	print(test.train())
