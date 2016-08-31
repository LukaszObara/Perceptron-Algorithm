'''Perceptron.py'''

#### Libraries ####
# Third Party Libraries 
import numpy as np

__version__ = "1.1"
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
	# location = 'C:\\Users\\Lukasz Obara\\OneDrive\\Documents\\' \
	#				+'Test Files\\gates.csv'

	# data = np.genfromtxt(location, delimiter=',')
	# all_gates = []

	# for i in data:
	# 	temp = np.append(i[:-1], 1)
	# 	foo = (np.array(temp, dtype=int), np.array([i[-1]], dtype=int))
	# 	all_gates.append(foo)

	# gates = [all_gates[i:i+4] for i in range(0, len(all_gates), 4)]
	# test = Perceptron(gates[0][0:4])
	# print(test.train())
	pass
