#!/usr/bin/python

import sys, random
import os
import time
from random import shuffle
import math, logging
from algorithms import compute_ada_grad_updates
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('gm_multi_out')
import layers as layers
import myutils as utils

# import pprint
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
import mmap
import cPickle
from operator import itemgetter
from collections import OrderedDict


theano.config.exception_verbosity='high'

config = {}


print 'loading config file', sys.argv[1]

config = utils.loadConfig(sys.argv[1])


networkfile = config['net']
num_of_hidden_units = int(config['hidden_units'])
trainfile=config['Etrain']
devfile=config['Edev']
targetTypesFile=config['typefile']

vectorFile=config['ent_vectors']

#learning parameters
learning_rate = float(config['lrate'])
batch_size = int(config['batchsize'])
n_epochs = int(config['nepochs'])
num_neg = int(config['numneg'])
l_reg = ''
l_weight = 0.000001
if 'loss_reg' in config:
    l_reg = config['loss_reg']
    l_weight = float(config['loss_weight'])
    
use_tanh_out = False
if 'tanh' in config:
    use_tanh_out = True  
outputtype = config['outtype'] #hinge or softmax
usetypecosine = False
if 'typecosine' in config:
    usetypecosine = utils.str_to_bool(config['typecosine'])
    
upto = -1
(t2ind, n_targets, wordvectors, vectorsize, typefreq_traindev) = utils.loadTypesAndVectors(targetTypesFile, vectorFile, -1)

(rvt, input_matrix_train, iet,resvectrnall, ntrn) = utils.fillOnlyEntityData(trainfile,vectorsize, wordvectors, t2ind, n_targets, upto=upto, binoutvec=True)
print "number of training examples:" + str(len(iet))

(rvd, input_matrix_dev, ied,resvecdevall, ntdev) = utils.fillOnlyEntityData(devfile,vectorsize, wordvectors, t2ind, n_targets, upto=upto, binoutvec=True)
print "number of validation examples:" +  str(len(ied))

if usetypecosine:
    print 'using cosine(e,t) as another input feature'
    typevecmatrix = utils.buildtypevecmatrix(t2ind, wordvectors, vectorsize) # a matrix with size: 102 * dim 
    e2simmatrix_train = utils.buildcosinematrix(input_matrix_train, typevecmatrix)
    e2simmatrix_dev = utils.buildcosinematrix(input_matrix_dev, typevecmatrix)
    input_matrix_train = utils.extend_in_matrix(input_matrix_train, e2simmatrix_train)
    input_matrix_dev = utils.extend_in_matrix(input_matrix_dev, e2simmatrix_dev)

rng = numpy.random.RandomState(23455)

dt = theano.config.floatX  # @UndefinedVariable
train_set_x = theano.shared(numpy.matrix(input_matrix_train, dtype=dt))  # @UndefinedVariable
valid_set_x = theano.shared(numpy.matrix(input_matrix_dev, dtype=dt))
train_set_y = theano.shared(numpy.matrix(resvectrnall, dtype=numpy.dtype(numpy.int32)))
valid_set_y = theano.shared(numpy.matrix(resvecdevall, dtype=numpy.dtype(numpy.int32)))

n_train_batches = train_set_x.get_value(borrow=True).shape[0]
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
n_train_batches /= batch_size
n_valid_batches /= batch_size

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')  # the data is presented as rasterized images
y = T.imatrix('y')  # the labels are presented as 1D vector of  multi
                        # [int] labels
######################
# BUILD ACTUAL MODEL #
######################
logger.info('... building the model')
rng = numpy.random.RandomState(23455)
layer1 = layers.HiddenLayer(rng, input=x, n_in=input_matrix_train.shape[1], n_out=num_of_hidden_units, activation=T.tanh)

outlayers = []
cost = 0.
out_errors = []
total_errs = 0
params = layer1.params
for i in range(n_targets):
    oneOutLayer = layers.OutputLayer(input=layer1.output, n_in=num_of_hidden_units, n_out=2)
    onelogistic = layers.SoftmaxLoss(input=oneOutLayer.score_y_given_x, n_in=2, n_out=2)
    params += oneOutLayer.params
    outlayers.append(oneOutLayer)
    total_errs += onelogistic.errors(y[:,i])
    cost += onelogistic.negative_log_likelihood(y[:,i])

# total_errors 
total_errs /= n_targets

updates = compute_ada_grad_updates(cost, params, learning_rate)

train_model = theano.function([index], cost, updates=updates,
              givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
                })
# 
validate_model = theano.function([index], total_errs, 
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

###############
# TRAIN MODEL #
logger.info('... training')
# early-stopping parameters
improvement_threshold = 0.9995  # a relative improvement of this much is
validation_frequency = n_train_batches

best_params = []
best_validation_loss = numpy.inf
best_iter = 0
test_score = 0.
start_time = time.clock()
val_losses = []
max_val_loss = int(math.ceil(n_epochs / 4.))
epoch = 0
done_looping = False

possible_indecis = [i for i in xrange(n_train_batches)]
while (epoch < n_epochs) and (not done_looping):
        shuffle(possible_indecis)
        epoch = epoch + 1
        print 'epoch = ', epoch
        for minibatch_index in xrange(n_train_batches):
            random_index = possible_indecis[minibatch_index] 
            iter = (epoch - 1) * n_train_batches + minibatch_index + 1
            
            if iter % 1000  == 0:
                print 'training @ iter = ', iter

            # print train_set_y.get_value(borrow=True).shape
            
            cost_ij = train_model(random_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = []
                
                for i in xrange(n_valid_batches):
#                     print str(i)
                    validation_losses.append(validate_model(i))
#                 validation_losses = [validate_model(i) for i
#                                      in xrange(n_valid_batches)]
 
                this_validation_loss = numpy.mean(validation_losses)
                logger.info('epoch %i, iteration %i, validation cost %f , train cost %f ' % \
                      (epoch, iter, this_validation_loss, cost_ij))

                if this_validation_loss < utils.minimal_of_list(val_losses):
                    del val_losses[:]
                    val_losses.append(this_validation_loss)
                    best_iter = iter
                    best_validation_loss = this_validation_loss
                    best_params = [[layer1.params[0].get_value(borrow=False), layer1.params[1].get_value(borrow=False)]]
                    for j in range(n_targets):
                        best_params.append([outlayers[j].params[0].get_value(borrow=False), outlayers[j].params[1].get_value(borrow=False)])
                    print('**best results updated! waiting for %i more validations!', max_val_loss)

                elif len(val_losses) < max_val_loss:
                    logger.info('addinig new validation to the val_losses, len(val_lossses) is %d', len(val_losses))
                    val_losses.append(this_validation_loss)
                    if len(val_losses) == max_val_loss:
                        done_looping = True
                        break
    
end_time = time.clock()
logger.info('Optimization complete.')
logger.info('Best validation score of %f %% obtained at iteration %i, %%', best_validation_loss * 100., best_iter + 1)
print >> sys.stderr, ('The code for file ' + 
                          os.path.split(__file__)[1] + 
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
logger.info('Saving net.')
save_file = open(networkfile, 'wb')
cPickle.dump(best_params[0][0], save_file, -1)
cPickle.dump(best_params[0][1], save_file, -1)
for i in range(n_targets):
    cPickle.dump(best_params[i + 1][0], save_file, -1)
    cPickle.dump(best_params[i + 1][1], save_file, -1)
save_file.close()
