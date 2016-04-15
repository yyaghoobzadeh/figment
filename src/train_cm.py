#!/usr/bin/python

import sys
import os
import time
# import pprint
import numpy
import theano
import theano.tensor as T
import mmap
import cPickle
from operator import itemgetter
from collections import OrderedDict
from random import shuffle
import logging
import math
import layers as layers
import myutils as utils

theano.config.exception_verbosity='high'
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('train_nn_new.py')
dt = theano.config.floatX  # @UndefinedVariable

config = {}


config = utils.loadConfig(sys.argv[1])
logger.info('training with configs = ', config )

sampleddir = config['sampled_dir'] 
networkfile = config['net']
num_of_hidden_units = int(config['hidden_units'])
nkerns=[int(config['numberfilters'])]
trainfile=sampleddir + config['trainsetfile']
devfile = sampleddir + config['devexample']
targetTypesFile=config['typefile']

vectorFile=config['vectors']
filtersize = int(config['filtersize'])

#learning parameters
learning_rate = float(config['lrate'])
batch_size = int(config['batchsize'])
n_epochs = int(config['nepochs'])

num_neg = int(config['numneg'])
useSum = True
sumval = config['useSum']
if 'False' in sumval:
    useSum = False
l_reg = ''
l_weight = 0.00001
if 'loss_reg' in config:
    l_reg = config['loss_reg']
    l_weight = float(config['loss_weight'])
    
use_tanh_out = False
if 'tanh' in config:
    use_tanh_out = True     

outputtype = config['outtype'] #hinge or softmax
leftsize = int(config['left'])
rightsize = int(config['right'])
sum_window = int(config['sumwindow'])

label_type = config['target'] #nt or ct

max_window = int(config['maxwindow'])
slotposition = max_window / 2
logger.info('slotpostion is %d', slotposition)
(typeIndMap, n_targets, wordvecs, vectorsize, typefreq_traindev) = utils.loadTypesAndVectors(targetTypesFile, vectorFile)

(traincontexts,resultVectorTrain, resultVectorTrainAll) = utils.read_lines_data(trainfile, typeIndMap, max_window, label_type, -1)
logger.info("number of training examples: %d", len(traincontexts))
inputMatrixTrain = utils.adeltheanomatrix_flexible(slotposition, vectorsize, traincontexts, wordvecs, leftsize, rightsize, sum_window, useSum)

(contextlistDev,resultVectorDev,resultVectorDevAll) = utils.read_lines_data(devfile, typeIndMap, max_window, label_type, -1)
logger.info("number of validation examples: %d", len(contextlistDev))

inputMatrixDev = utils.adeltheanomatrix_flexible(slotposition, vectorsize, contextlistDev, wordvecs, leftsize, rightsize, sum_window, useSum)
contextsize = leftsize + rightsize + 1


#build negative results for hinge loss

resultMatrixDevAll = numpy.empty(shape=(len(resultVectorDevAll), n_targets))
for i in xrange(len(resultVectorDevAll)):
    for j in range(n_targets):
        resultMatrixDevAll[i][j] = resultVectorDevAll[i][j]


# print baseline performance
sorted_typefreq_traindev = OrderedDict(sorted(typefreq_traindev.items(), key=itemgetter(1), reverse=True))
mostFreqType = sorted_typefreq_traindev.keys()[0]
c = utils.getNumberOftypeInset(resultVectorDev, typeIndMap[mostFreqType])
prec_bl = c / float(len(resultVectorDev))
logger.info('number of most freq type in devset: ' + str(c) + ' for type: ' + mostFreqType)
logger.info('the most freq baseline for notable type prediction has classification error: ' + str(1.0 - prec_bl))  

# train network
rng = numpy.random.RandomState(23455)

dt = dt
train_set_x = theano.shared(numpy.matrix(inputMatrixTrain, dtype=dt))
valid_set_x = theano.shared(numpy.matrix(inputMatrixDev, dtype=dt))
train_set_y = theano.shared(numpy.matrix(resultVectorTrainAll, dtype=numpy.dtype(numpy.int32)))
valid_set_y = theano.shared(numpy.matrix(resultVectorDevAll, dtype=numpy.dtype(numpy.int32)))

valid_set_y_all = theano.shared(numpy.matrix(resultMatrixDevAll, dtype=numpy.dtype(numpy.int32)))

ishape = [vectorsize, contextsize]  # this is the size of context matrizes

# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow=True).shape[0]
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
n_train_batches /= batch_size
n_valid_batches /= batch_size

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')  # the data is presented as rasterized images
y = T.imatrix('y')  # the labels are presented as 1D vector of
                        # [int] labels
# y_all = T.ivector('y_all')
######################
# BUILD ACTUAL MODEL #
######################
logger.info('... building the model')


layer2_input = x.reshape((batch_size, 1, ishape[0], ishape[1])).flatten(2)

# construct a fully-connected sigmoidal layer
layer2 = layers.HiddenLayer(rng, input=layer2_input, n_in=ishape[0] * ishape[1],
                         n_out=num_of_hidden_units, activation=T.tanh)


outlayers = []
cost = 0.
out_errors = []
total_errs = 0
params = layer2.params

logger.info('n_in in each softmax: %d and n_out: %d', num_of_hidden_units, 2)
for i in range(n_targets):
    oneOutLayer = layers.OutputLayer(input=layer2.output, n_in=num_of_hidden_units, n_out=2)
#     oneOutLayer = MyLogisticRegression(input=layer1.output, n_in=num_of_hidden_units, n_out=2)
    onelogistic = layers.SoftmaxLoss(input=oneOutLayer.score_y_given_x, n_in=2, n_out=2)
    params += oneOutLayer.params
    outlayers.append(oneOutLayer)
    total_errs += onelogistic.errors(y[:,i])
    cost += onelogistic.negative_log_likelihood(y[:,i])
    
# total_errors 
total_errs /= n_targets

# the cost we minimize during training is the NLL of the model

# L2_reg = T.sum(allweights** 2)
# if l_reg != '' and l_reg == 'L2':
#     cost += l_weight * L2_reg
    

validate_model = theano.function([index], total_errs,
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

# grads = T.grad(cost, params)
# 
# updates = []
# for param_i, grad_i in zip(params, grads):
#         updates.append((param_i, param_i - learning_rate * grad_i))
logger.info('Using adagrad...')
accumulator=[]
for para_i in params:
    eps_p = numpy.zeros_like(para_i.get_value(borrow=True), dtype=dt)
    accumulator.append(theano.shared(eps_p, borrow=True))
          
grads = T.grad(cost, params)
updates = []
for param_i, grad_i, acc_i in zip(params, grads, accumulator):
    acc = acc_i + T.sqr(grad_i)
    updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc) + 1e-10)))  # AdaGrad
    updates.append((acc_i, acc))    


train_model = theano.function([index], cost, updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        })
    

###############
# TRAIN MODEL #
###############
logger.info('training started ...')
# early-stopping parameters
patience = 4*len(traincontexts)/batch_size  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                           # found
improvement_threshold = 0.9995  # a relative improvement of this much is
                                 # considered significant
validation_frequency = min(n_train_batches, patience / 2)
logger.info('validation freq: %d', validation_frequency)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

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
        logger.info('epoch = %i, iter = %i', epoch, iter)
        
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            rnd_index = possible_indecis[minibatch_index]
            if iter % 100000  == 0:
                logger.info('training @ iter = %d', iter)

            # print train_set_y.get_value(borrow=True).shape

            cost_ij = train_model(rnd_index)
            if epoch == 1:
                continue
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
 
                this_validation_loss = numpy.mean(validation_losses) 
                logger.info('epoch %i, iteration %i, validation cost %f , train cost %f ' % \
                      (epoch, iter, this_validation_loss, cost_ij))

                if this_validation_loss < utils.minimal_of_list(val_losses):
                    del val_losses[:]
                    val_losses.append(this_validation_loss)
                    best_iter = iter
                    best_params = [[layer2.params[0].get_value(borrow=False), layer2.params[1].get_value(borrow=False)]]
                    for j in range(n_targets):
                        best_params.append([outlayers[j].params[0].get_value(borrow=False), outlayers[j].params[1].get_value(borrow=False)])
                    best_validation_loss = this_validation_loss
                    logger.info('**best results updated! waiting for %i more validations!', max_val_loss)

                elif len(val_losses) < max_val_loss:
                    logger.info('addinig new validation to the val_losses, len(val_lossses) is %i', len(val_losses))
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
logger.info('model saved in %s', save_file)


