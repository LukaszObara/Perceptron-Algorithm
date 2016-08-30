#### Libraries ####
# Third Party Libraries 
import numpy as np

class Perceptron(object):
	threshold = 0.5
	learning_rate = 0.1

	def __init__(self, data):
		self.data = data
		self.weights = np.zeros(len(data[0][0]))

	def train(self):
		j = 0

		while True:
			error_count = 0

			for input_vector, result_vector in self.data:
				result = np.dot(input_vector, self.weights) \
						  > Perceptron.threshold
				error = result_vector - result

				if error != 0:
					error_count += 1
					self.weights += input_vector * error \
									 * Perceptron.learning_rate

			if error_count == 0 or j == 200:
				print('Automatic Stop --- j = 200')
				break

			j += 1

		return self.weights

class Sigmoid(object):
	def activation_fn(self, z):
		"""The sigmoid function.
		"""
		return 1.0/(1.0+np.exp(-z))

	def prime(self, z):
		"""The derivative of the sigmoid function.
		"""
		return self.activation_fn(z)*(1-self.activation_fn(z))


class Network(object):
	def __init__(self, sizes, neurons=Sigmoid):
		self.layer_num = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(y, x)
						for x, y in zip(self.sizes[:-1], self.sizes[1:])]
		self.neurons = neurons()

	def feedforward(self, a):
		"""Return the output of the network if "a" is input. If 
		"normalize" is set to true then the outputs will be normalized 
		with the softmax function.
		"""
		for b, w in zip(self.biases, self.weights):
			a = self.neurons.activation_fn(np.dot(w, a)+b)

		return a


if __name__ == '__main__':
	# We mimicking the MNIST data set, noting that the data is arranged
	# as a list containg a 2-tuple of arrays, that is 
	# [(np.array(), np.array()), (np.array(), np.array()), ...]

	# nand =  [(np.array([0, 0, 1]), np.array([0])), 
	# 		 (np.array([1, 0, 1]), np.array([1])), 
	# 		 (np.array([0, 1, 1]), np.array([1])), 
	# 		 (np.array([1, 1, 1]), np.array([0]))]  

	# test = Perceptron(nand)
	# print(test.train())

	nand_sigmoid =  [(np.array([0, 0]), np.array([0])), 
					 (np.array([1, 0]), np.array([1])), 
					 (np.array([0, 1]), np.array([1])), 
					 (np.array([1, 1]), np.array([0]))]

	net = Network([2, 4, 1])
	print(net.feedforward(nand_sigmoid[0][0]))
	# print(net.weights)

	

