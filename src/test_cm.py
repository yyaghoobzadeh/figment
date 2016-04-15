#!/usr/bin/python

import cPickle
from collections import OrderedDict
import mmap
from operator import itemgetter
import os
import sys
import time

from numpy import ndarray, savetxt, set_printoptions, vstack
import numpy
import theano

import theano.tensor as T
import layers as layers
import myutils as utils
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('test_nn_new.py')


# import pprint
config = utils.loadConfig(sys.argv[1])
logger.info(config )
testsetfile = sys.argv[2]
outfile = sys.argv[3]

sampleddir = config['sampled_dir']
networkfile = config['net']
num_of_hidden_units = int(config['hidden_units'])
learning_rate = float(config['lrate'])
batch_size = int(config['batchsize']) * 5
nkerns=[int(config['numberfilters'])]
targetTypesFile=config['typefile']
filtersize = int(config['filtersize'])
vectorFile=config['vectors']
num_neg=int(config['numneg'])
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
max_window = int(config['maxwindow'])
slotposition = max_window / 2

label_type = config['target'] #nt or ct

(typeIndMap, n_targets, wordvecs, vectorsize, typefreq_traindev) = utils.loadTypesAndVectors(targetTypesFile, vectorFile, -1)
logger.info('word2vec vectors are loaded')
(testcontexts,resultVectorTest, resultVectorTestAll) = utils.read_lines_data(testsetfile, typeIndMap, max_window, label_type, -1)
logger.info("number of test examples: %d", len(testcontexts))

inputMatrixTest = utils.adeltheanomatrix_flexible(slotposition, vectorsize, testcontexts, wordvecs, leftsize, rightsize, sum_window, useSum)
contextsize = leftsize + rightsize + 1
################# for memory ############
testcontexts = []; wordvecs = []; 
##################### the network #######################
test_set_x = theano.shared(numpy.matrix(inputMatrixTest, dtype=theano.config.floatX))  # @UndefinedVariable
test_set_y = theano.shared(numpy.matrix(resultVectorTestAll, dtype=numpy.dtype(numpy.int32)))

rng = numpy.random.RandomState(23455)

n_test_batches = test_set_x.get_value(borrow=True).shape[0]
n_test_batches /= batch_size

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')  # the data is presented as rasterized images
y = T.imatrix('y')  # the labels are presented as 1D vector of
                        # [int] labels
ishape = [vectorsize, contextsize]  # this is the size of context matrizes
# print "vocabsize: " + str(vocabsize)
filtersize = [1, filtersize]  # filter height, filter width
pool = [1, ishape[1] - filtersize[1] + 1] 


# layer2_input = x.reshape((batch_size, 1, ishape[0], ishape[1])).flatten(2)

layer2 = layers.HiddenLayer(rng, input=x, n_in=ishape[0] * ishape[1],
                         n_out=num_of_hidden_units, activation=T.tanh)


# classify the values of the fully-connected sigmoidal layer
outlayers = []
cost = 0.
out_errors = []
total_errs = 0
predicted_probs = []
logger.info('n_in in softmax_layer: %d and n_out: %d', num_of_hidden_units, 2)

for i in range(n_targets):
    oneOutLayer = layers.OutputLayer(input=layer2.output, n_in=num_of_hidden_units, n_out=2)
#     oneOutLayer = MyLogisticRegression(input=layer1.output, n_in=num_of_hidden_units, n_out=2)
    onelogistic = layers.SoftmaxLoss(input=oneOutLayer.score_y_given_x, n_in=2, n_out=2)
    outlayers.append(oneOutLayer)
    one_unit_probs = onelogistic.getOutProbs()
    predicted_probs.append(one_unit_probs)

test_predicted_probs = theano.function([index], predicted_probs,
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size]})

# load parameters
netfile = open(networkfile)
layer2.params[0].set_value(cPickle.load(netfile), borrow=False)
layer2.params[1].set_value(cPickle.load(netfile), borrow=False)
for i in range(n_targets):
    outlayers[i].params[0].set_value(cPickle.load(netfile), borrow=False)
    outlayers[i].params[1].set_value(cPickle.load(netfile), borrow=False)

# test net on test file

f = open(outfile, 'w')
outs= ''
logger.info('saving the results...')
for i in range(n_test_batches + 1):
    itemlist = test_predicted_probs(i) #itemlist is the list of matrixes
    print len(itemlist) #
    for j in range(batch_size):
        for t_ind in range(n_targets):
            f.write(str(itemlist[t_ind][j][1]) + ' ')
        f.write('\n')
f.close()    
logger.info('test results saved in: %s', outfile)
