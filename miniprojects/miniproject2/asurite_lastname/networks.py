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
		
		# NN Params
		self.weights = []
		self.biases = []
		self.lr = 0.05

		# Saved states for backprop
		self.prev_outs = []
		self.weighted_sums = []

		# Input layer
		self.weights.append(np.array(np.random.rand(self.n_trn_smp_dim, self.n_lyr_nodes)))
		self.biases.append(np.array(np.random.rand(self.n_lyr_nodes)))
		self.prev_outs.append(np.zeros(self.n_trn_smp_dim))
		self.weighted_sums.append(np.zeros(self.n_lyr_nodes))

		# Hidden Layer(s)
		for lyr in range(0, self.n_hidden_lyr):
			self.weights.append(np.array(np.random.rand(self.n_lyr_nodes, self.n_lyr_nodes)))
			self.biases.append(np.array(np.random.rand(self.n_lyr_nodes)))
			self.prev_outs.append(np.zeros(self.n_lyr_nodes))
			self.weighted_sums.append(np.zeros(self.n_lyr_nodes))

		# Output Layer
		self.weights.append(np.array(np.random.rand(self.n_lyr_nodes, self.n_outputs)))
		self.biases.append(np.array(np.random.rand(self.n_outputs)))
		self.prev_outs.append(np.zeros(self.n_lyr_nodes))
		self.weighted_sums.append(np.zeros(self.n_outputs))

		self.params = []  # [(w,b),(w,b)]
		for lyr in range(len(self.weights)):
			self.params.append((self.weights[lyr], self.biases[lyr]))

		# Start Training
		for iter in range(100):
			output = self.forward(self.x)
			#print("Output: ")
			#print(output.shape)
			#print(output)	
			#print("Labels: ")
			#print(self.y.shape)
			#print(self.y)
			#print("Error: " )
			err = self.error(output, self.y)
			#print(err.shape)
			print(err)	
			self.backward(err)
	
	def activation_sigmoid (self, t):
		""" 
		Method that gives the response of a sigmoid to an input 

		Returns:
			Gives a floating point output of s(t), s() being the sigmoid function 
		"""
		return (1/(1+np.exp(-t)))
	
	def activation_sigmoid_d (self, t):
		""" 
		Method that gives the DERIVATIVE of the sigmoid 

		Returns:
			Gives a floating point output of s'(t), s() being the sigmoid function 
		"""
		s_t = self.activation_sigmoid(t)
		return (s_t*(1-s_t))

	def forward (self, data_x): 	
		""" 
		Method that returns the output of a pass through the NN 

		Returns:
			Gives a numpy.ndarray of the same size of the input. The array consists of class labels 
		"""

		# Vectorize the activation function so we can apply it to the vector
		self.activation_sigmoid = np.vectorize(self.activation_sigmoid, otypes=[np.float])

		
		prev_lyr = data_x
		#print("-------Input Layer--------")
		#print(prev_lyr)
		for lyr in range(len(self.weights)):
			self.prev_outs[lyr] = prev_lyr  # KEEP THIS HERE - order here is very important
			#print("Layer: " + str(lyr))
			#print(prev_lyr.shape)
			weighted_sum = np.add(np.dot(prev_lyr, self.weights[lyr]), self.biases[lyr])
			self.weighted_sums[lyr] = weighted_sum
			prev_lyr = self.activation_sigmoid(weighted_sum)
			#print("Mult: " + str(mult.shape) + " Biases: " + str(self.biases[lyr].shape) + " Prev Layer: " + str(prev_lyr.shape))
			#print("Layer: " + str(lyr))

		return prev_lyr

	def error (self, nn_output, label):
		""" 
		Method that returns the mean squared error.  

		Returns:
			Gives a scalar of the mean squared error. 
			The error is calculated by taking the sum of 1/2 the squared difference between nn_output and label
			error = (1/N)SUM_i(0.5(label_i - nn_output_i))
		"""
		difference = np.subtract(np.transpose(label), np.transpose(nn_output))
		sqrd = np.dot(difference, np.transpose(difference))
		err = np.multiply(sqrd, 1.0/(2.0*len(label)))
		return err
		
		

	def backward (self, cur_error):
		""" 
		Method that performs backwards propogation of the error gradient, updating the weights 

		Returns:
			Nothing
		"""
		# Vectorize the derivative of the activation function so we can apply it to the vector
		self.activation_sigmoid_d = np.vectorize(self.activation_sigmoid_d, otypes=[np.float])

		# Iterate backwards through layers
		for lyr in range(len(self.weights)-1, -1, -1):
			d_weightedsum_weights = self.prev_outs[lyr]
			d_prevouts_weightedsum = self.activation_sigmoid_d(self.weighted_sums[lyr])
			d_err_prevouts = cur_error #THIS NEEDS TO BE GENERALIZED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			#print(d_weightedsum_weights.shape)
			#print(d_prevouts_weightedsum.shape)
			#print(d_err_prevouts.shape)
			d_err_weights = np.multiply(np.dot(np.transpose(d_weightedsum_weights), d_prevouts_weightedsum), d_err_prevouts)
			#print(self.weights[lyr].shape)
			#print(d_err_weights.shape)
			self.weights[lyr] = self.weights[lyr] - np.multiply(self.lr, d_err_weights)
			
  
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
