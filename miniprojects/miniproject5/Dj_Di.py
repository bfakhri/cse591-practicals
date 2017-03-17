import matplotlib
matplotlib.use('Agg')
import os
#os.environ['THEANO_FLAGS'] = "force_device=True, device=gpu, floatX=float32, exception_verbosity=high"
#os.environ['THEANO_FLAGS'] = "force_device=True, device=gpu, floatX=float32"
import theano
print("Theano Default Device: ")
print(theano.config.device)
import sys	# For CLI args
from yann.network import network
net = network()

dataset_params  = { "dataset": sys.argv[2], "id": 'mnist', "n_classes" : 10 }

# Begin ConvNet Setup using old params
# Load already trained params
from yann.utils.pickle import load
parts = sys.argv[1].split('/')
load_str = 'network'+parts[1]+'.pkl'
print "Loading Network From: " + load_str
old_params = load(load_str)
print "Old Layers: " 
print old_params.keys()
                    
net.add_layer(  type = "input", 
		id ="input", 
		dataset_init_args = dataset_params)


net.add_layer ( type = "conv_pool",
                origin = "input",
                id = "conv_pool_1",
                num_neurons = 20,
                filter_size = (5,5),
                pool_size = (2,2),
                activation = ('maxout', 'maxout', 2),
                batch_norm = True,
                regularize = True,
                verbose = True,
                init_params = old_params['conv_pool_1'],
		learnable = False)

net.add_layer ( type = "conv_pool",
                origin = "conv_pool_1",
                id = "conv_pool_2",
                num_neurons = 50,
                filter_size = (3,3),
                pool_size = (2,2),
                activation = ('maxout', 'maxout', 2),
                batch_norm = True,
                regularize = True,
                verbose = True,
                init_params = old_params['conv_pool_2'],
		learnable = False)

net.add_layer ( type = "dot_product",
		origin ="conv_pool_2",
		id = "dot_product_1",
		num_neurons = 800,
		regularize = True,
		activation ='relu',
		dropout_rate=0.5,
                init_params = old_params['dot_product_1'],
		learnable = False)

net.add_layer ( type = "dot_product",
		origin ="dot_product_1",
		id = "dot_product_2",
		num_neurons = 800,
		regularize = True,
		activation ='relu',
		dropout_rate=0.5, 
                init_params = old_params['dot_product_2'],
		learnable = False)


net.add_layer ( type = "classifier",
                id = "softmax",
                origin = "dot_product_2",
                num_classes = 10,
                activation = 'softmax',
		#init_params = old_params['softmax'],
		learnable = True)

net.add_layer ( type = "objective",
                id = "nll",
                origin = "softmax",
                )


optimizer_params =  {
            "momentum_type"       : 'nesterov',
            "momentum_params"     : (0.9, 0.95, 20),
            "regularization"      : (0.0001, 0.0001),
            "optimizer_type"      : 'rmsprop',
            "id"                  : 'bij'
                    }

net.add_module ( type = 'optimizer', params = optimizer_params )


learning_rates = (0.05, 0.01, 0.001)

net.cook( optimizer = 'bij',
          objective_layer = 'nll',
          datastream = 'mnist',
          classifier = 'softmax',
          )

net.train( epochs = (8, 8),
           validate_after_epochs = 1,
           training_accuracy = True,
           learning_rates = learning_rates,
           show_progress = True,
           early_terminate = True)

net.test()




