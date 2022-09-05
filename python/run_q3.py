import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from nn import *

#train_data = scipy.io.loadmat('../data/nist36_train.mat')
train_data = scipy.io.loadmat(r"C:\Users\sahar\Desktop\Acads\CVB-Spring22\hw5-1\hw5\data\nist36_train.mat")
valid_data = scipy.io.loadmat(r"C:\Users\sahar\Desktop\Acads\CVB-Spring22\hw5-1\hw5\data\nist36_valid.mat")
test_data = scipy.io.loadmat(r"C:\Users\sahar\Desktop\Acads\CVB-Spring22\hw5-1\hw5\data\nist36_test.mat")

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

#print(train_x.shape, train_y.shape) #10800, 1024 and 10800,36 (one-hot encoded)
#print(valid_x.shape, valid_y.shape) #3600 eg
#print(test_x.shape, test_y.shape) #1800 eg

np.random.seed(1)

if False: # view the data
    np.random.shuffle(train_x)
    for crop in train_x:
        plt.imshow(crop.reshape(32,32).T, cmap="Greys")
        plt.show()

max_iters = 200
# pick a batch size, learning rate
batch_size = 128
learning_rate = 0.001
hidden_size = 64
##########################
##### your code here #####
##########################


batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers
initialize_weights(train_x.shape[1], hidden_size, params, "layer1")
initialize_weights(hidden_size, train_y.shape[1], params, "output")
layer1_W_initial = np.copy(params["Wlayer1"]) # copy for Q3.3

train_loss = []
valid_loss = []
train_acc = []
valid_acc = []
for itr in range(max_iters):
    # # record training and validation loss and accuracy for plotting
    # h1 = forward(train_x,params,'layer1')
    # probs = forward(h1,params,'output',softmax)
    # loss, acc = compute_loss_and_acc(train_y, probs)
    # train_loss.append(loss/train_x.shape[0])
    # train_acc.append(acc)
    # h1 = forward(valid_x,params,'layer1')
    # probs = forward(h1,params,'output',softmax)
    # loss, acc = compute_loss_and_acc(valid_y, probs)
    # valid_loss.append(loss/valid_x.shape[0])
    # valid_acc.append(acc)

    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################
        h1 = forward(xb,params,'layer1',sigmoid)
        prob = forward(h1,params,'output',softmax)
        #print(prob.shape)
        #print(1, xb.shape, yb.shape, prob.shape)
        loss, acc = compute_loss_and_acc(yb, prob)
        # loss
        # be sure to add loss and accuracy to epoch totals
        total_loss += loss
        total_acc += acc
        
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
    
    total_acc = total_acc/len(batches)
    total_loss = total_loss/len(batches)
    train_loss.append(total_loss)
    train_acc.append(total_acc)
    
    #validation
    h1 = forward(valid_x,params,'layer1')
    probs = forward(h1,params,'output',softmax)
    loss, acc = compute_loss_and_acc(valid_y, probs)
    valid_loss.append(loss/valid_x.shape[0])
    valid_acc.append(acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

# record final training and validation accuracy and loss
# h1 = forward(train_x,params,'layer1')
# probs = forward(h1,params,'output',softmax)
# loss, acc = compute_loss_and_acc(train_y, probs)
# train_loss.append(loss/train_x.shape[0])
# train_acc.append(acc)
# h1 = forward(valid_x,params,'layer1')
# probs = forward(h1,params,'output',softmax)
# loss, acc = compute_loss_and_acc(valid_y, probs)
# valid_loss.append(loss/valid_x.shape[0])
# valid_acc.append(acc)

# report validation accuracy; aim for 75%
print('Validation accuracy: ', valid_acc[-1])

# compute and report test accuracy
h1 = forward(test_x,params,'layer1')
test_probs = forward(h1,params,'output',softmax)
_, test_acc = compute_loss_and_acc(test_y, test_probs)
print('Test accuracy: ', test_acc)

# save the final network
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
        pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# plot loss curves
plt.plot(range(len(train_loss)), train_loss, label="training")
plt.plot(range(len(valid_loss)), valid_loss, label="validation")
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(train_loss)-1)
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.show()

# plot accuracy curves
plt.plot(range(len(train_acc)), train_acc, label="training")
plt.plot(range(len(valid_acc)), valid_acc, label="validation")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.xlim(0, len(train_acc)-1)
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.show()


# Q3.3

# visualize weights
fig = plt.figure(figsize=(8,8))
plt.title("Layer 1 weights after initialization")
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.05)
for i, ax in enumerate(grid):
    ax.imshow(layer1_W_initial[:,i].reshape((32, 32)).T, cmap="Greys")
    ax.set_axis_off()
plt.show()

v = np.max(np.abs(params['Wlayer1']))
fig = plt.figure(figsize=(8,8))
plt.title("Layer 1 weights after training")
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.05)
for i, ax in enumerate(grid):
    ax.imshow(params['Wlayer1'][:,i].reshape((32, 32)).T, cmap="Greys", vmin=-v, vmax=v)
    ax.set_axis_off()
plt.show()


# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
#print(train_y.shape) #eg, out - 10k,36
# compute confusion matrix
##########################
##### your code here #####
##########################

#test_y and test_probs used ## true in rows and pred in columns
#print(test_y.shape, test_probs.shape)

for i in range(test_probs.shape[0]):
    row = np.argmax(test_y[i])
    col = np.argmax(test_probs[i])
    confusion_matrix[row][col] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid()
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.xlabel("predicted label")
plt.ylabel("true label")
plt.colorbar()
plt.show()