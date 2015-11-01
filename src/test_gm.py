#!/usr/bin/python

import os
import sys, random
import time

import cPickle
from collections import OrderedDict
import mmap
from operator import itemgetter


import numpy
import theano
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import theano.tensor as T

import src.classification.nn.model.layers as layers
import src.classification.common.myutils as utils


theano.config.exception_verbosity='high'

print 'loading config file'
config = utils.loadConfig(sys.argv[1])


networkfile = config['net']
num_of_hidden_units = int(config['hidden_units'])

targetTypesFile=config['typefile']

vectorFile=config['ent_vectors']
contextsize = 1#int(config['contextsize'])

#learning parameters
learning_rate = float(config['lrate'])
batch_size = 1#int(config['batchsize'])
n_epochs = int(config['nepochs'])
num_neg = int(config['numneg'])

testfile=sys.argv[2]
outf=sys.argv[3]
use_tanh_out = False
outputtype = config['outtype'] #hinge or softmax
usetypecosine = False
if 'typecosine' in config:
    usetypecosine = utils.str_to_bool(config['typecosine'])
    
(t2ind, n_targets, wordvectors, vectorsize, typefreq_traindev) = utils.loadTypesAndVectors(targetTypesFile, vectorFile)

(rvt, input_matrix_test, iet,resvectstnall, ntrn) = utils.fillOnlyEntityData(testfile,vectorsize, wordvectors, t2ind, n_targets, upto=-1, ds='test', binoutvec=True)

# train network
rng = numpy.random.RandomState(23455)
if usetypecosine:
    print 'using cosine(e,t) as another input feature'
    typevecmatrix = utils.buildtypevecmatrix(t2ind, wordvectors, vectorsize) # a matrix with size: 102 * dim 
    e2simmatrix_test = utils.buildcosinematrix(input_matrix_test, typevecmatrix)
    input_matrix_test = utils.extend_in_matrix(input_matrix_test, e2simmatrix_test)

dt = theano.config.floatX  # @UndefinedVariable

index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')  # the data is presented as rasterized images
y = T.imatrix('y')  # the labels are presented as 1D vector of
                        # [int] labels
######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'
rng = numpy.random.RandomState(23455)
layer1 = layers.HiddenLayer(rng, input=x, n_in=input_matrix_test.shape[1],n_out=num_of_hidden_units, activation=T.tanh)

outlayers = []
cost = 0.
out_errors = []
predicted_probs = []

for i in range(n_targets):
    oneOutLayer = layers.OutputLayer(input=layer1.output, n_in=num_of_hidden_units, n_out=2)
    onelogistic = layers.SoftmaxLoss(input=oneOutLayer.score_y_given_x, n_in=2, n_out=2)
    outlayers.append(oneOutLayer)
    one_unit_probs = onelogistic.getOutProbs()
    predicted_probs.append(one_unit_probs)
    
# total_errors 
# total_errs /= n_targets

netfile = open(networkfile)
layer1.params[0].set_value(cPickle.load(netfile), borrow=False)
layer1.params[1].set_value(cPickle.load(netfile), borrow=False)
for i in range(n_targets):
    outlayers[i].params[0].set_value(cPickle.load(netfile), borrow=False)
    outlayers[i].params[1].set_value(cPickle.load(netfile), borrow=False)
netfile.close()

print "number of testing examples:" + str(len(iet))

# inputMatrixTest = obj.adeltheanomatrix(vectorsize, contextlistTest, wordvectors, 0, contextsize)
test_set_x = theano.shared(numpy.matrix(input_matrix_test, dtype=theano.config.floatX))  # @UndefinedVariable


n_test_batches = test_set_x.get_value(borrow=True).shape[0]

test_predicted_probs = theano.function([index], predicted_probs,
             givens={
                 x: test_set_x[index * batch_size: (index + 1) * batch_size]})

f = open(outf, 'w')
print 'saving the results...'
for i in range(n_test_batches):
    f.write(iet[i] + '\t')
    for item in test_predicted_probs(i):
        f.write(str(item[0][1]) + ' ')
    f.write('\n')
f.close()

# print str(t / len(resultVectorTest))
print 'results are saved in ', outf







