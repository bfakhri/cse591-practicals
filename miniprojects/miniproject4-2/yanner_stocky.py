import os
#os.environ['THEANO_FLAGS'] = "force_device=True, device=gpu, lib.cnmem=0.75, floatX=float32, mode=DebugMode,DebugMode.check_py=False"
#os.environ['THEANO_FLAGS'] = "force_device=True, device=gpu, lib.cnmem=0.45, floatX=float32"
os.environ['THEANO_FLAGS'] = "force_device=True, device=gpu, floatX=float32"
import theano
print("Theano Default Device: ")
print(theano.config.device)
import sys	# For CLI args
from yann.network import network
net = network()

dataset_params  = { "dataset": "_datasets/_dataset_72566", "id": 'mnist', "n_classes" : 10 }

# Begin ConvNet Setup
net.add_layer(type = "input", id ="input", dataset_init_args = dataset_params)

net.add_layer ( type = "conv_pool",
                origin = "input",
                id = "conv_pool_1",
                num_neurons = 64,
                filter_size = (15,15),
                pool_size = (3,3),
                activation = ('maxout', 'maxout', 2),
                batch_norm = True,
                regularize = True,
                verbose = True
            )

net.add_layer (type = "dot_product",
               origin ="conv_pool_1",
               id = "dot_product_1",
               num_neurons = 512,
               regularize = True,
               activation ='relu')

net.add_layer (type = "dot_product",
               origin ="dot_product_1",
               id = "dot_product_2",
               num_neurons = 128,
               regularize = True,
               activation ='relu')

net.add_layer ( type = "classifier",
                id = "softmax",
                origin = "dot_product_2",
                num_classes = 10,
                activation = 'softmax',
                )

net.add_layer ( type = "objective",
                id = "nll",
                origin = "softmax",
                )

optimizer_params =  {
            "momentum_type"       : 'polyak',
            "momentum_params"     : (0.9, 0.95, 20),
            #"regularization"      : (0.1, 0.2),
            #"regularization"      : (0.3, 0.4),
            #"regularization"      : (0.001, 0.002),
            "regularization"      : (0.01, 0.2),
            "optimizer_type"      : 'rmsprop',
            #"optimizer_type"      : 'adagrad',
            "id"                  : 'bij'
                    }
net.add_module ( type = 'optimizer', params = optimizer_params )


#learning_rates = (0.05, 0.01, 0.001)
learning_rates = (0.05, 0.007, 0.00001)

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
                
