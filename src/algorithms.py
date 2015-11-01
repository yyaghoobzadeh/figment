'''
Created on Nov 1, 2015

@author: yy1
'''
import numpy, theano
import theano.tensor as T

def compute_ada_grad_updates(cost, params, learning_rate):
    dt = theano.config.floatX  # @UndefinedVariable
    accumulator=[]
    for para_i in params:
        eps_p = numpy.zeros_like(para_i.get_value(borrow=True), dtype=dt)
        accumulator.append(theano.shared(eps_p, borrow=True))
              
            # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc) + 1e-10)))  # AdaGrad
        updates.append((acc_i, acc))      
    return updates  