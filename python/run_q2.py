from turtle import back
import numpy as np
# you should write your functions in nn.py
from nn import *
from util import *


# fake data
# feel free to plot it in 2D
# what do you think these 4 classes are?
g0 = np.random.multivariate_normal([3.6,40],[[0.05,0],[0,10]],10)
g1 = np.random.multivariate_normal([3.9,10],[[0.01,0],[0,5]],10)
g2 = np.random.multivariate_normal([3.4,30],[[0.25,0],[0,5]],10)
g3 = np.random.multivariate_normal([2.0,10],[[0.5,0],[0,10]],10)
x = np.vstack([g0,g1,g2,g3])
# we will do XW + B
# that implies that the data is N x D

# create labels
y_idx = np.array([0 for _ in range(10)] + [1 for _ in range(10)] + [2 for _ in range(10)] + [3 for _ in range(10)])
# turn to one_hot
y = np.zeros((y_idx.shape[0],y_idx.max()+1))
y[np.arange(y_idx.shape[0]),y_idx] = 1

# parameters in a dictionary
params = {}

# Q 2.1
# initialize a layer
initialize_weights(2,25,params,'layer1')
initialize_weights(25,4,params,'output')
assert(params['Wlayer1'].shape == (2,25))
assert(params['blayer1'].shape == (25,))

# expect 0, [0.05 to 0.12]
print("{}, {:.2f}".format(params['blayer1'].mean(),params['Wlayer1'].std()**2))
print("{}, {:.2f}".format(params['boutput'].mean(),params['Woutput'].std()**2))

# Q 2.2.1
# implement sigmoid
test = sigmoid(np.array([-1000,1000]))
print('should be zero and one\t',test.min(),test.max())
# implement forward
h1 = forward(x,params,'layer1')
print(h1.shape)
# Q 2.2.2
# implement softmax
probs = forward(h1,params,'output',softmax)
# make sure you understand these values!
# positive, ~1, ~1, (40,4)
print(probs.min(),min(probs.sum(1)),max(probs.sum(1)),probs.shape)

# Q 2.2.3
# implement compute_loss_and_acc
loss, acc = compute_loss_and_acc(y, probs)
# should be around -np.log(0.25)*40 [~55] or higher, and 0.25
# if it is not, check softmax!
print("{}, {:.2f}".format(loss,acc))

# here we cheat for you
# the derivative of cross-entropy(softmax(x)) is probs - 1[correct actions]
delta1 = probs - y

# we already did derivative through softmax
# so we pass in a linear_deriv, which is just a vector of ones
# to make this a no-op
delta2 = backwards(delta1,params,'output',linear_deriv)
# Implement backwards!
backwards(delta2,params,'layer1',sigmoid_deriv)

# W and b should match their gradients sizes
for k,v in sorted(list(params.items())):
    if 'grad' in k:
        name = k.split('_')[1]
        print(name,v.shape, params[name].shape)

# Q 2.4
batches = get_random_batches(x,y,5)
# print batch sizes
print([_[0].shape[0] for _ in batches])
batch_num = len(batches)

# WRITE A TRAINING LOOP HERE
max_iters = 500
learning_rate = 1e-3
# with default settings, you should get loss < 35 and accuracy > 75%
for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0
    for xb,yb in batches:
        ##########################
        ##### your code here #####
        ##########################
        # forward
        h1 = forward(xb,params,'layer1',sigmoid)
        prob = forward(h1,params,'output',softmax)
        #print(prob.shape)
        #print(1, xb.shape, yb.shape, prob.shape)
        loss, acc = compute_loss_and_acc(yb, prob)
        # loss
        # be sure to add loss and accuracy to epoch totals
        total_loss += loss
        avg_acc += acc
        
        # backward
        delta1 = prob-yb
        # apply gradient 
        delta2 = backwards(delta1, params,'output',linear_deriv)
        delta3 = backwards(delta2, params, 'layer1', sigmoid_deriv)
        # gradients should be summed over batch samples

        params['blayer1'] = params['blayer1']-(learning_rate*params['grad_blayer1'])
        params['boutput'] = params['boutput']-(learning_rate*params['grad_boutput'])
        params['Wlayer1'] = params['Wlayer1']-(learning_rate*params['grad_Wlayer1'])
        params['Woutput'] = params['Woutput']-(learning_rate*params['grad_Woutput'])

    avg_acc /= len(batches)

        
    if itr % 100 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))


# Q 2.5 should be implemented in this file
# you can do this before or after training the network. 

# compute gradients using forward and backward
h1 = forward(x,params,'layer1')
probs = forward(h1,params,'output',softmax)
loss, acc = compute_loss_and_acc(y, probs)
delta1 = probs - y
delta2 = backwards(delta1,params,'output',linear_deriv)
backwards(delta2,params,'layer1',sigmoid_deriv)

# save the old params
import copy
params_orig = copy.deepcopy(params)

# compute gradients using finite difference
eps = 1e-6
#print(params.items())
for k,v in params.items():
    if '_' in k: 
        continue
    # we have a real parameter!
    # print(k) #wlayer, blayer, wout,bout
    #print(v.shape) # 2,25 ... 25 ...25,4 ... 4
    if len(v.shape)==2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                copy_params_minus = copy.deepcopy(params)
                copy_params_minus[k][i,j] = v[i,j]-eps
                copy_params_plus = copy.deepcopy(params)
                copy_params_plus[k][i,j] = v[i,j]+eps

                h1_minus = forward(x,copy_params_minus,'layer1')
                probs_minus = forward(h1_minus,copy_params_minus,'output',softmax)
                loss_minus, acc_minus = compute_loss_and_acc(y,probs_minus)

                h1_plus = forward(x,copy_params_plus,'layer1')
                probs_plus = forward(h1_plus,copy_params_plus,'output',softmax)
                loss_plus, acc_plus = compute_loss_and_acc(y,probs_plus)

                params['grad_'+k][i,j] = (loss_plus-loss_minus)/(2*eps)

    else:
        for i in range(v.shape[0]):
            copy_params_minus = copy.deepcopy(params)
            copy_params_minus[k][i] = v[i]-eps
            copy_params_plus = copy.deepcopy(params)
            copy_params_plus[k][i] = v[i]+eps

            h1_minus = forward(x,copy_params_minus,'layer1')
            probs_minus = forward(h1_minus,copy_params_minus,'output',softmax)
            loss_minus, acc_minus = compute_loss_and_acc(y,probs_minus)

            h1_plus = forward(x,copy_params_plus,'layer1')
            probs_plus = forward(h1_plus,copy_params_plus,'output',softmax)
            loss_plus, acc_plus = compute_loss_and_acc(y,probs_plus)

            params['grad_'+k][i] = (loss_plus-loss_minus)/(2*eps)

total_error = 0
for k in params.keys():
    if 'grad_' in k:
        # relative error
        err = np.abs(params[k] - params_orig[k])/np.maximum(np.abs(params[k]),np.abs(params_orig[k]))
        err = err.sum()
        print('{} {:.2e}'.format(k, err))
        total_error += err
# should be less than 1e-4
print('total {:.2e}'.format(total_error))
