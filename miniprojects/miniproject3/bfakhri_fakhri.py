#import os
#os.environ['THEANO_FLAGS'] = "force_device=True, device=gpu"
#import theano
#print("Theano Default Device: ")
#print(theano.config.device)
import sys	# For CLI args
from yann.network import network
net = network()
dataset_params  = { "dataset": sys.argv[1], "id": 'mnist', "n_classes" : 10 }
net.add_layer(type = "input", id ="input", dataset_init_args = dataset_params)

net.add_layer (type = "dot_product",
               origin ="input",
               id = "dot_product_1",
               num_neurons = 800,
               regularize = True,
               activation ='relu')

net.add_layer (type = "dot_product",
               origin ="dot_product_1",
               id = "dot_product_2",
               num_neurons = 800,
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
            #"momentum_type"       : 'nesterov',
            "momentum_type"       : 'polyak',
            #"momentum_params"     : (0.9, 0.95, 25),
            "momentum_params"     : (0.9, 0.95, 30),
            #"regularization"      : (0.1, 0.2),
            "regularization"      : (0.01, 0.02),
            "optimizer_type"      : 'rmsprop',
            #"optimizer_type"      : 'adagrad',
            "id"                  : 'bij'
                    }
net.add_module ( type = 'optimizer', params = optimizer_params )


learning_rates = (0.05, 0.01, 0.001)

net.cook( optimizer = 'bij',
          objective_layer = 'nll',
          datastream = 'mnist',
          classifier = 'softmax',
          )

net.train( epochs = (20, 20),
           validate_after_epochs = 2,
           training_accuracy = True,
           learning_rates = learning_rates,
           show_progress = True,
           early_terminate = True)

net.test()
                
