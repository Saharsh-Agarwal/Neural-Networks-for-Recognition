import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import string

train_data = scipy.io.loadmat('data/nist36_train.mat')
valid_data = scipy.io.loadmat('data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter() #gfg

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################
## print(train_x.shape) # 10800, 1024
initialize_weights(train_x.shape[1], hidden_size, params, 'inp')
initialize_weights(hidden_size, hidden_size, params, 'hidden1')
initialize_weights(hidden_size, hidden_size, params, 'hidden2')
initialize_weights(hidden_size, train_x.shape[1], params, 'output')

#print(params.elements())
#print(params.keys())

wnb_names = [i for i in params.keys()]
for k in wnb_names:
    params['init_'+k] = np.zeros(params[k].shape) #updates of weights as 0 const

#print(params.keys())

# should look like your previous training loops
losses = []
#prob_prev = None
for itr in range(max_iters):
    total_loss = 0
    loss = 0
    for xb,_ in batches:
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing

        ##########################
        ##### your code here #####
        ##########################
         # forward
        h1 = forward(xb,params,'inp',relu)
        h2 = forward(h1,params,'hidden1',relu)
        h3 = forward(h2,params,'hidden2',relu)
        prob = forward(h3,params,'output',sigmoid)
        #print(prob.shape)
        #if prob_prev==None or prob.all()!=prob_prev.all():
        #    prob_prev=prob
        #    print("yes")

        loss = ((xb-prob)**2).sum()
        total_loss += loss

        # backward
        delta1 = -2*(xb-prob)
        # apply gradient 
        delta2 = backwards(delta1, params,'output',sigmoid_deriv)
        delta3 = backwards(delta2, params, 'hidden2', relu_deriv)
        delta4 = backwards(delta3, params, 'hidden1', relu_deriv)
        backwards(delta4, params, 'inp', relu_deriv)
        # gradients should be summed over batch samples

        for l in ['output','hidden2','hidden1','inp']:
            params['init_W' + l] = 0.9*params['init_W' + l] - learning_rate * params['grad_W' + l]
            params['W' + l] += params['init_W' + l] 
            params['init_b' + l] = 0.9*params['init_b' + l]  - learning_rate * params['grad_b' + l]
            params['b' + l]+= params['init_b' + l]
        
    
    losses.append(total_loss/train_x.shape[0])
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# plot loss curve
plt.plot(range(len(losses)), losses)
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(losses)-1)
plt.ylim(0, None)
plt.grid()
plt.show()


# Q5.3.1
# choose 5 labels (change if you want)
visualize_labels = ["A", "B", "C", "1", "2"]

# get 2 validation images from each label to visualize
visualize_x = np.zeros((2*len(visualize_labels), valid_x.shape[1]))
for i, label in enumerate(visualize_labels):
    idx = 26+int(label) if label.isnumeric() else string.ascii_lowercase.index(label.lower())
    choices = np.random.choice(np.arange(100*idx, 100*(idx+1)), 2, replace=False)
    visualize_x[2*i:2*i+2] = valid_x[choices]

# run visualize_x through your network
# name the output reconstructed_x
##########################
##### your code here #####
##########################
#print(visualize_x.shape)
#print(xb.shape)
h1 = forward(visualize_x,params,'inp',relu)
h2 = forward(h1,params,'hidden1',relu)
h3 = forward(h2,params,'hidden2',relu)
reconstructed_x = forward(h3,params,'output',sigmoid)

# visualize
fig = plt.figure()
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(len(visualize_labels), 4), axes_pad=0.05)
for i, ax in enumerate(grid):
    if i % 2 == 0:
        ax.imshow(visualize_x[i//2].reshape((32, 32)).T, cmap="Greys")
    else:
        ax.imshow(reconstructed_x[i//2].reshape((32, 32)).T, cmap="Greys")
    ax.set_axis_off()
plt.show()


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
##########################
h1 = forward(valid_x,params,'inp',relu)
h2 = forward(h1,params,'hidden1',relu)
h3 = forward(h2,params,'hidden2',relu)
ans = forward(h3,params,'output',sigmoid)
psn = 0
for i in range(valid_x.shape[0]):
    psn += peak_signal_noise_ratio(valid_x[i],ans[i])

psnr = psn/valid_x.shape[0]
print("PSNR Value", psnr)
