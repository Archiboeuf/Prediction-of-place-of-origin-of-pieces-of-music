# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:44:50 2018

@author: benji
"""

from __future__ import absolute_import
from __future__ import print_function

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from builtins import range
import pandas as pd
import matplotlib.pyplot as plt
import folium
import random as rd

path='C:\\Users\\benji\\Desktop\\Cours\\Stochastic Optimization and automatic differentiation\\Geographical Original of Music\\Geographical Original of Music\\'
df=pd.read_csv(path+'default_plus_chromatic_features_1059_tracks.txt',header=None)
df=df.T.drop_duplicates().reset_index(drop=True).T #removing duplicated columns


df[72]=df[72]/90 #transforms latitude into a number between -1 and 1
df[73]=df[73]/180 #transforms longitude into a number between -1 and 1
pi=np.pi

def arccosine(long1,lat1,long2,lat2): #great-circle distance (long and lat given between [-1,1])
    true_long1,true_long2=long1*pi,long2*pi
    true_lat1,true_lat2=lat1*pi/2,lat2*pi/2    
    return(np.arcsin(np.sqrt(np.sin((true_lat2-true_lat1)/2)**2
        +np.cos(true_lat1)*np.cos(true_lat2)*np.sin((true_long2-true_long1)/2)**2)))

#splitting train and test bases
rd.seed(0)
N,M=df.shape
train_size=0.8
id_train=rd.sample(range(N),int(train_size*N))
var=df[df.columns[:-2]]
coord=df[df.columns[-2:]]
var_train=var[var.index.isin(id_train)]
var_test=var[~var.index.isin(id_train)]
coord_train=coord[coord.index.isin(id_train)]
coord_test=coord[~coord.index.isin(id_train)]

#defining how to create a neural network
class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.N = 0

    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

def make_batches(N_total, N_batch):
    start = 0
    batches = []
    while start < N_total:
        batches.append(slice(start, start + N_batch))
        start += N_batch
    return batches

def make_nn_funs(input_shape, layer_specs, L2_reg):
    parser = WeightsParser()
    cur_shape = input_shape
    for layer in layer_specs:
        N_weights, cur_shape = layer.build_weights_dict(cur_shape)
        parser.add_weights(layer, (N_weights,))

    def predictions(W_vect, inputs):
        """Outputs normalized log-probabilities.
        shape of inputs : [data, color, y, x]"""
        cur_units = inputs
        for layer in layer_specs:
            cur_weights = parser.get(W_vect, layer)
            cur_units = layer.forward_pass(cur_units, cur_weights)
        return cur_units

    def loss(W_vect, X, T):
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        pred=predictions(W_vect, X)
        dist=np.array([arccosine(pred[i][0],pred[i][1],T[i][0],T[i][1]) for i in range(len(pred))])
        dist = np.sum(dist)
        return - log_prior + dist

    def frac_err(W_vect, X, T):
        pred=predictions(W_vect, X)
        dist=np.array([arccosine(pred[i][0],pred[i][1],T[i][0],T[i][1]) for i in range(len(pred))])
        return np.mean(dist)

    return parser.N, predictions, loss, frac_err

class full_layer(object):
    def __init__(self, size,nonlinearity):
        self.size = size
        self.nonlinearity=nonlinearity

    def build_weights_dict(self, input_shape):
        # Input shape is anything (all flattened)
        input_size = np.prod(input_shape, dtype=int)
        self.parser = WeightsParser()
        self.parser.add_weights('params', (input_size, self.size))
        self.parser.add_weights('biases', (self.size,))
        return self.parser.N, (self.size,)

    def forward_pass(self, inputs, param_vector):
        params = self.parser.get(param_vector, 'params')
        biases = self.parser.get(param_vector, 'biases')
        if inputs.ndim > 2:
            inputs = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))
        return self.nonlinearity(np.dot(inputs[:, :], params) + biases)

# process data
reshaping = lambda x : x.reshape((x.shape[0], 1, x.shape[1]))
train_musics, train_labels, test_musics, test_labels = (
        np.array(var_train),np.array(coord_train), np.array(var_test), np.array(coord_test))
train_musics = reshaping(train_musics) 
test_musics  = reshaping(test_musics)  
N_data = train_musics.shape[0]

#cross-validation
test_perf_optim=10**6
for L2_reg in [10**i for i in range(-3,3)]: #regularization L2
    for neurone_number in [10,30,50,100,200]: #number of neurones in intermediate layer
        for learning_rate in [0.05,0.001]:
            print('NEW PARAMETERS')
            print(L2_reg,neurone_number,learning_rate)
            # Make neural net functions
            # Network parameters
            input_shape = (1, M-2)
            layer_specs = [full_layer(neurone_number,lambda x : np.tanh(x)),
                           full_layer(2,lambda x : np.tanh(x))]
            # Training parameters
            param_scale = 0.1
            momentum = 0.9
            batch_size = 32
            num_epochs = 200
            
            N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(input_shape, layer_specs, L2_reg)
            loss_grad = grad(loss_fun)
            
            # Initialize weights
            rs = npr.RandomState()
            W = rs.randn(N_weights) * param_scale
            
            print("    Epoch      |    Train err  |   Test error  ")
            def print_perf(epoch, W):
                test_perf  = frac_err(W, test_musics, test_labels)
                train_perf = frac_err(W, train_musics, train_labels)
                print("{0:15}|{1:15}|{2:15}".format(epoch, train_perf, test_perf))
            
            # Train with sgd
            batch_idxs = make_batches(N_data, batch_size)
            cur_dir = np.zeros(N_weights)
            
            for epoch in range(num_epochs):
                print_perf(epoch, W)
                for idxs in batch_idxs:
                    grad_W = loss_grad(W, train_musics[idxs], train_labels[idxs])
                    cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_W
                    W -= learning_rate * cur_dir
            
            test_perf  = frac_err(W, test_musics, test_labels)
            if test_perf<test_perf_optim: #updating optimal paramaters
                print('NEW OPTIMUM')
                print(L2_reg,neurone_number,learning_rate)
                test_perf_optim=test_perf
                params_optim=L2_reg,neurone_number,learning_rate
                W_optim=W
                
            
print(params_optim) #(1, 10, 0.001)
print(W_optim) 
print(test_perf_optim) #0.248408857468
print(test_perf_optim*2*6371) # mean error in km : 3165
#saving results
pd.DataFrame(np.array(params_optim),index=['L2_reg','neurone_number','learning_rate']).to_csv(path+'params_optim coordonnees.csv',header=False)
pd.DataFrame(W_optim,).to_csv(path+'W_optim coordonnees.csv',header=False)

#charging results
#params_optim=np.array(pd.read_csv(path+'params_optim coordonnees.csv',header=None,index_col=0)[1])
#W_optim=np.array(pd.read_csv(path+'W_optim coordonnees.csv',header=None,index_col=0)[1])

#visualisation of results
L2_reg,neurone_number,learning_rate=params_optim
neurone_number=int(neurone_number) #neurone_number is a float if parameters have been loaded
layer_specs = [full_layer(neurone_number,lambda x : np.tanh(x)),
                           full_layer(2,lambda x : np.tanh(x))]
N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(input_shape, layer_specs, L2_reg)

# checking error
pred=pred_fun(W_optim,test_musics)
dist=pd.DataFrame([arccosine(pred[i][0],pred[i][1],test_labels[i][0],test_labels[i][1]) for i in range(len(pred))],columns=['Error distance'])
dist['Error distance']=dist['Error distance']*2*6371 #get distance in km
dist.describe()

dist.hist()
plt.savefig(path+'hist error coordinates.png')    



#creation of a map with predictions and true locations
pred=pd.DataFrame(pred)
test_labels=pd.DataFrame(test_labels)

#get back on the true ranges of coordinates
pred[0]=pred[0]*90 
pred[1]=pred[1]*180
test_labels[0]=test_labels[0]*90
test_labels[1]=test_labels[1]*180
            
m = folium.Map() #creation of a map

colors=['red', 'blue', 'green', 'purple', 'orange', 'darkred',
        'lightred', 'pink', 'darkblue', 'darkgreen']
#        , 'cadetblue', 'darkpurple', 'white', 'beige', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
n_col=len(colors)

#sampling n_col musics to display prediction and true location on the map
samp=rd.sample(range(len(pred)),n_col) 
for j in range(n_col):
    i=samp[j]
    folium.Marker(pred.loc[i], popup='pred '+str(i),icon=folium.Icon(color=colors[j],icon='info-sign')).add_to(m)
    folium.Marker(test_labels.loc[i], popup='test '+str(i),icon=folium.Icon(color=colors[j],icon='cloud')).add_to(m)

m.save('map.html')
m




########################################
# With heatmaps instead of coordinates #
########################################

heatmap=pd.read_csv(path+'heatmap.txt',header=None,sep='   ')

coord=heatmap
coord_train=coord[coord.index.isin(id_train)]
coord_test=coord[~coord.index.isin(id_train)]

#redefining NN with another loss function
def make_nn_funs(input_shape, layer_specs, L2_reg):
    parser = WeightsParser()
    cur_shape = input_shape
    for layer in layer_specs:
        N_weights, cur_shape = layer.build_weights_dict(cur_shape)
        parser.add_weights(layer, (N_weights,))

    def predictions(W_vect, inputs):
        """Outputs normalized log-probabilities.
        shape of inputs : [data, color, y, x]"""
        cur_units = inputs
        for layer in layer_specs:
            cur_weights = parser.get(W_vect, layer)
            cur_units = layer.forward_pass(cur_units, cur_weights)
        return cur_units

    def loss(W_vect, X, T):
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        pred=predictions(W_vect, X)
        exp_kl=(T/pred)**T #exp(Kullback Leibler) (to avoid problems in log)
        kl = np.mean(np.sum(np.log(exp_kl),axis=1)) 
        return - log_prior + kl

    def frac_err(W_vect, X, T):
        pred=predictions(W_vect, X)
        exp_kl=(T/pred)**T #exp(Kullback Leibler) (to avoid problems in log)
        kl = np.mean(np.sum(np.log(exp_kl),axis=1)) 
        return kl

    return parser.N, predictions, loss, frac_err

#process data
reshaping = lambda x : x.reshape((x.shape[0], 1, x.shape[1]))
train_musics, train_labels, test_musics, test_labels = (
        np.array(var_train),np.array(coord_train), np.array(var_test), np.array(coord_test))
train_musics = reshaping(train_musics) 
test_musics  = reshaping(test_musics)  
N_data = train_musics.shape[0]

softmax=lambda x : np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)
#cross-validation
test_perf_optim=10**6
for L2_reg in [0]+[10**i for i in range(-3,3)]:
    for neurone_number in [10,50,100,500]:
        for learning_rate in [0.05,0.001]:
            print('NEW PARAMETERS')
            print(L2_reg,neurone_number,learning_rate)
            # Make neural net functions
            # Network parameters
            input_shape = (1, M-2)
            layer_specs = [full_layer(neurone_number,lambda x : np.tanh(x)),
                           full_layer(400,softmax)]
            # Training parameters
            param_scale = 0.1
            momentum = 0.9
            batch_size = 32
            num_epochs = 200
            
            N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(input_shape, layer_specs, L2_reg)
            loss_grad = grad(loss_fun)
            
            # Initialize weights
            rs = npr.RandomState()
            W = rs.randn(N_weights) * param_scale
            
            # Check the gradients numerically, just to be safe
            # quick_grad_check(loss_fun, W, (train_musics[:50], train_labels[:50]))
            
            print("    Epoch      |    Train err  |   Test error  ")
            def print_perf(epoch, W):
                test_perf  = frac_err(W, test_musics, test_labels)
                train_perf = frac_err(W, train_musics, train_labels)
                print("{0:15}|{1:15}|{2:15}".format(epoch, train_perf, test_perf))
            
            # Train with sgd
            batch_idxs = make_batches(N_data, batch_size)
            cur_dir = np.zeros(N_weights)
            
            for epoch in range(num_epochs):
                print_perf(epoch, W)
                for idxs in batch_idxs:
                    grad_W = loss_grad(W, train_musics[idxs], train_labels[idxs])
                    cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_W
                    W -= learning_rate * cur_dir
            
            test_perf  = frac_err(W, test_musics, test_labels)
            print(test_perf)
            if test_perf<test_perf_optim:
                print('NEW OPTIMUM')
                print(L2_reg,neurone_number,learning_rate)
                test_perf_optim=test_perf
                params_optim=L2_reg,neurone_number,learning_rate
                W_optim=W
                
                
print(params_optim) #0.001 100 0.05
print(W_optim) 
print(test_perf_optim) #1.5971829534570983
#saving results
pd.DataFrame(np.array(params_optim),index=['L2_reg','neurone_number','learning_rate']).to_csv(path+'params_optim heatmap.csv',header=False)
pd.DataFrame(W_optim,).to_csv(path+'W_optim heatmap.csv',header=False)

#checking error
pred=pred_fun(W_optim,test_musics)
exp_kl=(test_labels/pred)**test_labels
kl = pd.Series(np.sum(np.log(exp_kl),axis=1)) 
kl.describe()
kl.hist()
plt.savefig(path+'hist error heatmap.png')    

# ################################################################################################
# We train deeper the neural network with the best parameters identified in this cross-validation
# ################################################################################################

L2_reg,neurone_number,learning_rate=params_optim
layer_specs = [full_layer(neurone_number,lambda x : np.tanh(x)),
                           full_layer(400,softmax)]
N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(input_shape, layer_specs, L2_reg)
W = W_optim

print("    Epoch      |    Train err  |   Test error  ")
def print_perf(epoch, W):
    test_perf  = frac_err(W, test_musics, test_labels)
    train_perf = frac_err(W, train_musics, train_labels)
    print("{0:15}|{1:15}|{2:15}".format(epoch, train_perf, test_perf))

# Train with sgd
batch_idxs = make_batches(N_data, batch_size)
cur_dir = np.zeros(N_weights)

for epoch in range(1000):
    print_perf(epoch, W)
    for idxs in batch_idxs:
        grad_W = loss_grad(W, train_musics[idxs], train_labels[idxs])
        cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_W
        W -= learning_rate * cur_dir

test_perf  = frac_err(W, test_musics, test_labels)
if test_perf<test_perf_optim:
    print('NEW OPTIMUM')
    print(L2_reg,neurone_number,learning_rate)
    test_perf_optim=test_perf
    params_optim=L2_reg,neurone_number,learning_rate
    W_optim=W
    
print(test_perf) #1.556313289128265
train_perf = frac_err(W, train_musics, train_labels)
print(train_perf) #0.70796363153031783 maybe some overfit as the train error is half the test error (L2 reg too small?)
#saving weights
pd.DataFrame(W_optim,).to_csv(path+'W_optim heatmap highly trained.csv',header=False)

#checking error
pred=pred_fun(W_optim,test_musics)
exp_kl=(test_labels/pred)**test_labels #exp(Kullback Leibler) (to avoid problems in log)
kl = pd.Series(np.sum(np.log(exp_kl),axis=1)) 
kl.describe()
kl.hist()
plt.savefig(path+'hist error heatmap highly trained.png')    


#Creation of maps
pred=pd.DataFrame(pred)
test_labels=pd.DataFrame(test_labels)

def labels_to_heatmap(labels): #labels : array of length 400 (20 x 20 heatmap)
    hm=[]
    for i,lab in enumerate(labels):
        lat=-90+(0.5+i%20)*180/20
        long=-180+(0.5+i//20)*360/20
        hm.append([lat,long,lab])
    return(hm)

samp=rd.sample(range(len(pred)),10)
for i in samp:
    temp=pred.loc[i]
    hm=labels_to_heatmap(temp)
    m=folium.Map()
    folium.plugins.HeatMap(hm,min_opacity=0,max_val=max([x[2] for x in hm]),max_zoom=2).add_to(m)
    m.save(path+'heatmap pred {}.html'.format(i))
    
    temp=test_labels.loc[i]
    hm=labels_to_heatmap(temp)
    m=folium.Map()
    folium.plugins.HeatMap(hm,min_opacity=0,max_val=max([x[2] for x in hm]),max_zoom=2).add_to(m)
    m.save(path+'heatmap true {}.html'.format(i))
    
    
    
#we set bigger L2 reg to have less overfit (maybe there were not enough iterations in the cross
#validation in order to identify the best parameters when we want to train more the model)
    
L2_reg,neurone_number,learning_rate=params_optim
L2_reg=0.01
layer_specs = [full_layer(neurone_number,lambda x : np.tanh(x)),
                           full_layer(400,softmax)]
N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(input_shape, layer_specs, L2_reg)
W = W_optim

print("    Epoch      |    Train err  |   Test error  ")
def print_perf(epoch, W):
    test_perf  = frac_err(W, test_musics, test_labels)
    train_perf = frac_err(W, train_musics, train_labels)
    print("{0:15}|{1:15}|{2:15}".format(epoch, train_perf, test_perf))

# Train with sgd
batch_idxs = make_batches(N_data, batch_size)
cur_dir = np.zeros(N_weights)

for epoch in range(2000):
    print_perf(epoch, W)
    for idxs in batch_idxs:
        grad_W = loss_grad(W, train_musics[idxs], train_labels[idxs])
        cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_W
        W -= learning_rate * cur_dir

test_perf  = frac_err(W, test_musics, test_labels)
print(test_perf)#2.4933215681296366
train_perf = frac_err(W, train_musics, train_labels) #2.399404667192811
#both errors are similar (no overfit) but they are higher than the errors with the previous L2 reg
pd.DataFrame(W,).to_csv(path+'W not optim heatmap highly trained.csv',header=False)

pred=pred_fun(W_optim,test_musics)
pred=pd.DataFrame(pred)

for i in samp:
    temp=pred.loc[i]
    hm=labels_to_heatmap(temp)
    m=folium.Map()
    folium.plugins.HeatMap(hm,min_opacity=0,max_val=max([x[2] for x in hm]),max_zoom=2).add_to(m)
    m.save(path+'heatmap pred {} not optimal.html'.format(i))

    
    