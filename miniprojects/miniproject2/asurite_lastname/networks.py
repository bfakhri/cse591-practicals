#sample_submission.py
import numpy as np

class xor_net(object):
	"""
	This is a sample class for miniproject 2.

	Args:
		data: Is a tuple, ``(x,y)``
			  ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
			  data and data is spread along axis 1. If the array had only one dimension, it implies
			  that data is 1D.
			  ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.	
						  
	"""
	def __init__(self, data, labels):
		self.x = data
		self.y = labels
		self.n_trn_smp = data.shape[0]	
		self.n_trn_smp_dim = data.shape[1]
		self.n_hidden_lyr = 1
		self.n_lyr_nodes = 2
		self.n_outputs = 1
		
		# For debugging
		print("Training w/ " + str(self.n_trn_smp) + " samples that are " + str(self.n_trn_smp_dim) + " dimensional")
		print("Training w/ " + str(self.n_hidden_lyr) + " Hidden Layers")
		print("")
		
		self.weights = []
		self.biases = []

		# Input layer
		self.weights.append(np.array(np.random.rand(self.n_trn_smp_dim, self.n_lyr_nodes)))
		self.biases.append(np.array(np.random.rand(self.n_lyr_nodes)))

		# Hidden Layer(s)
		for lyr in range(0, self.n_hidden_lyr):
			self.weights.append(np.array(np.random.rand(self.n_lyr_nodes, self.n_lyr_nodes)))
			self.biases.append(np.array(np.random.rand(self.n_lyr_nodes)))

		# Output Layer
		self.weights.append(np.array(np.random.rand(self.n_lyr_nodes, self.n_outputs)))
		self.biases.append(np.array(np.random.rand(self.n_outputs)))

		self.params = []  # [(w,b),(w,b)]
		for lyr in range(len(self.weights)):
			self.params.append((self.weights[lyr], self.biases[lyr]))

		# Start Training
		for iter in range(10):
			for sample in range(self.x.shape[0]):
				output = self.forward(self.x[sample])
				print("Output: ")
				print(output.shape)
				print(output)	
				print("Error: " )
				err = self.error(output, self.y[sample])
				print(err.shape)
				print(err)	
		

	def forward (self, data_x): 	
		""" 
		Method that returns the output of a pass through the NN 

		Returns:
			Gives a numpy.ndarray of the same size of the input. The array consists of class labels 
		"""
		
		prev_lyr = data_x
		#print("-------Input Layer--------")
		#print(prev_lyr)
		for lyr in range(len(self.weights)):
			prev_lyr = np.add(np.dot(prev_lyr, self.weights[lyr]), self.biases[lyr])
			#print("Layer: " + str(lyr))
			#print(prev_lyr)

		return prev_lyr

	def error (self, nn_output, label):
		difference = np.subtract(label, nn_output)
		return np.multiply(np.multiply(difference, difference), 0.5)
		
		

	def backward (self, data_y):
		""" 
		Method that performs backwards propogation of the error gradient, updating the weights 

		Returns:
			Nothing
		"""
  
	def get_params (self):
		""" 
		Method that should return the model parameters.

		Returns:
			tuple of numpy.ndarray: (w, b). 

		Notes:
			This code will return an empty list for demonstration purposes. A list of tuples of 
			weoghts and bias for each layer. Ordering should from input to outputt

		"""
		return self.params

	def get_predictions (self, x):
		"""
		Method should return the outputs given unseen data

		Args:
			x: array similar to ``x`` in ``data``. Might be of different size.

		Returns:	
			numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of 
							``x`` 
		Notes:
			Temporarily returns random numpy array for demonstration purposes.							  
		"""		   
		# Here is where you write a code to evaluate the data and produce predictions.
		return np.random.randint(low =0, high = 2, size = x.shape[0])

class mlnn(xor_net):
	"""
	At the moment just inheriting the network above. 
	"""
	def __init__ (self, data, labels):
		super(mlnn,self).__init__(data, labels)


if __name__ == '__main__':
	pass 
