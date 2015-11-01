"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
from myutils import debug_print
__docformat__ = 'restructedtext en'


import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

dt = theano.config.floatX  # @UndefinedVariable

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
 
        #tmp = T.cast(input, 'float64')
        #print "in Hidden layer: " + str(tmp)
        print "in Hidden layer: " + str(n_in) + " " + str(n_out)

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # dt so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=dt)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=dt)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]
class OutputLayer(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        self.n_out = n_out
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=dt),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=dt),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.score_y_given_x = (T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.score_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]
        
        
    def resultsScores(self):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        return [T.argmax(self.score_y_given_x, axis=1), self.score_y_given_x, T.max(self.score_y_given_x, axis=1)]

class LossLayer(object):
    def __init__(self, input, n_in, n_out):
        """ 
        :type n_neg: int
        :param n_neg: number of negative used in ranking loss

        """
        self.input = input
        self.n_in = n_in
        self.n_out = n_out
        self.y_pred = T.argmax(input, axis=1)

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))  # @UndefinedVariable
        # check if y is of the correct datatype
        ones = T.ones_like(y, dtype=numpy.int32)
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))

#             return T.mean(T.neq(ones, y_all[self.y_pred]))
        else:
            raise NotImplementedError()
    def output(self):
        return [T.argmax(self.input, axis=1), self.input, T.max(self.input, axis=1)]
    
    
# this is still buggy      
class RankingLoss(LossLayer):

    def __init__(self, input, n_in, n_out, n_neq):
        """ 
        :type n_neg: int
        :param n_neg: number of negative used in ranking loss

        """
        self.n_neq = n_neq
        super(RankingLoss, self).__init__(input, n_in, n_out)

    def hinge_loss(self, y, y_neq):
        # first I should remove softmax and output the scores
        # other things to do is to add negative random samples to x and to y! 
        # self.score_y_given_x 
        # mean(T.maximum(0, 1. - self.score_y_given_x[T.arange(y.shapre[0],y]) + self.score_y_given_x[T.arange(y.shapre[0],y_neq]))
#         zeros = theano.shared(value=numpy.zeros((y.shape[0],),dtype=numpy.int32))
#         if y.ndim != self.n_in:
#             raise TypeError('y should have the same shape as self.y_pred',
#                 ('y', target.type, 'y_pred', self.input.type))  # @UndefinedVariable
            
        zeros = numpy.zeros(shape=self.n_neq, dtype=numpy.int32)
        ones = numpy.ones(shape=self.n_neq, dtype=dt)  # @UndefinedVariable
#         ones = ones - 0.5
        #one_neg_y = random.sample(, n)
        y_rep = zeros + y
        rm = ones - self.input[zeros, y_rep] + self.input[zeros, y_neq]
        return T.maximum(zeros, rm).mean()
    
    def hinge_loss_batch(self, y):
        # first I should remove softmax and output the scores
        # other things to do is to add negative random samples to x and to y! 
        # self.score_y_given_x 
        # mean(T.maximum(0, 1. - self.score_y_given_x[T.arange(y.shapre[0],y]) + self.score_y_given_x[T.arange(y.shapre[0],y_neq]))
#         zeros = theano.shared(value=numpy.zeros((y.shape[0],),dtype=numpy.int32))
#         if y.ndim != self.n_in:
#             raise TypeError('y should have the same shape as self.y_pred',
#                 ('y', target.type, 'y_pred', self.input.type))  # @UndefinedVariable
            
        #ones = numpy.ones(shape=y.shape[0], dtype=dt)  # @UndefinedVariable
        margins = T.maximum(0., 1. - self.input[T.arange(y.shape[0])] * self.input[T.arange(y.shape[0]), y])
        #margins[T.arange(y.shape[0]),y] = 0.
        return T.mean(margins)
    
    
class SoftmaxLoss(LossLayer):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression
        """
        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(input)
#         self.p_y_given_x = debug_print(self.p_y_given_x, 'scores')
        super(SoftmaxLoss, self).__init__(input, n_in, n_out)

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) #neg log probability for target class  

    def cross_entropy_loss(self, y):
        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x[T.arange(y.shape[0])], y[T.arange(y.shape[0])])) #neg log probability for target class
    
    def output(self):
        return [T.argmax(self.p_y_given_x, axis=1), self.p_y_given_x, T.max(self.p_y_given_x, axis=1)]
        
    def getOutProbs(self):
        return self.p_y_given_x

class SigmoidLoss(LossLayer):
    """Multi-label sigmoid + crossentropy as loss function

    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression
        """
        # compute vector of class-membership probabilities in symbolic form
        self.s_y_given_x = T.nnet.sigmoid(input)
        self.s_y_given_x = debug_print(self.s_y_given_x, 'scores', False)
        super(SigmoidLoss, self).__init__(input, n_in, n_out)

    def cross_entropy_loss(self, y):
        return T.mean(T.nnet.binary_crossentropy(self.s_y_given_x[T.arange(y.shape[0])], y[T.arange(y.shape[0])])) #neg log probability for target class
    
    def output(self):
        return [T.argmax(self.s_y_given_x, axis=1), self.s_y_given_x, T.max(self.s_y_given_x, axis=1)]
        
    def getOutScores(self):
        return self.s_y_given_x

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=dt),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=dt),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


