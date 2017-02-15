#sample_submission.py
# Bijan Fakhri
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
		# Vectorize the activation function so we can apply it to the vector
		self.activation_sigmoid = np.vectorize(self.activation_sigmoid, otypes=[np.float])
		# Vectorize the derivative of the activation function so we can apply it to the vector
		self.activation_sigmoid_d = np.vectorize(self.activation_sigmoid_d, otypes=[np.float])

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
		self.lr = 0.01

		# Saved states for backprop
		self.neuron_in = []	# Inputs to the neurons
		self.neuron_out = []	# Neuronal activations

		# Input layer
		self.weights.append(np.array(np.random.rand(self.n_trn_smp_dim, self.n_lyr_nodes)*2-1.0))
		self.biases.append(np.array(np.random.rand(self.n_lyr_nodes)*2-1.0))
		self.neuron_in.append(np.zeros(self.n_lyr_nodes))
		self.neuron_out.append(np.zeros(self.n_lyr_nodes))

		# Hidden Layer(s)
		for lyr in range(0, self.n_hidden_lyr):
			self.weights.append(np.array(np.random.rand(self.n_lyr_nodes, self.n_lyr_nodes)*2-1.0))
			self.biases.append(np.array(np.random.rand(self.n_lyr_nodes)*2-1.0))
			self.neuron_in.append(np.zeros(self.n_lyr_nodes))
			self.neuron_out.append(np.zeros(self.n_lyr_nodes))

		# Output Layer
		self.weights.append(np.array(np.random.rand(self.n_lyr_nodes, self.n_outputs)*2-1.0))
		self.biases.append(np.array(np.random.rand(self.n_outputs)*2-1.0))
		self.neuron_in.append(np.zeros(self.n_outputs))
		self.neuron_out.append(np.zeros(self.n_outputs))

		self.params = []  # [(w,b),(w,b)]
		for lyr in range(len(self.weights)):
			self.params.append((self.weights[lyr], self.biases[lyr]))

		# Start Training
		for iter in range(100):
			output = self.forward(self.x)
			err = self.error(output, self.y)
			self.backward(err, self.y)
	
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

		c_neuron_out = np.array(np.zeros(1)) 
		for lyr in range(len(self.weights)):
			if(lyr == 0):
				c_neuron_in = np.add(np.dot(data_x, self.weights[lyr]), self.biases[lyr])
			else:
				c_neuron_in = np.add(np.dot(c_neuron_out, self.weights[lyr]), self.biases[lyr])
	
			self.neuron_in[lyr] = c_neuron_in
			if(lyr == (len(self.weights)-1)):
				c_neuron_out = c_neuron_in
			else:
				c_neuron_out = self.activation_sigmoid(c_neuron_in)

			self.neuron_out[lyr] = c_neuron_out

			
				
		return c_neuron_out 

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
		
		

	def backward (self, cur_error, y):
		""" 
		Method that performs backwards propogation of the error gradient, updating the weights 

		Returns:
			Nothing
		"""

		# Calculate first derivative error
		a = np.subtract(self.neuron_out[-1].flatten(), y)
		b = self.activation_sigmoid_d(self.neuron_out[-1])
		c = np.multiply(a.flatten(),b.flatten())
		d = np.dot(np.transpose(np.expand_dims(c, axis=1)), self.neuron_out[-2])

		self.weights[-1] = self.weights[-1] - np.transpose(np.multiply(self.lr,d))
  
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
		#og = np.random.randint(low =0, high = 2, size = x.shape[0])
		#print("OG Shape: ")
		#print(og.shape)
		# Here is where you write a code to evaluate the data and produce predictions.
		predictions = self.forward(x).flatten()
		predictions_bin = np.array(np.zeros(len(predictions)))
		for p in range(len(predictions)):
			if(predictions[p] > 0.5):
				predictions_bin[p] = 1
			else:
				predictions_bin[p] = 0
		#print("Prediction Shape: ")
		#print(predictions.shape)
		return predictions_bin

class mlnn(xor_net):
	"""
	At the moment just inheriting the network above. 
	"""
	def __init__ (self, data, labels):
		super(mlnn,self).__init__(data, labels)


if __name__ == '__main__':
	pass 
