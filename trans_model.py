#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 19:16:36 2018

@author: ififsun
"""

import torch
import torch.nn as nn
#import torch.utils.data as Data
#import torchvision
#import torch.nn.functional as F
from base_model import base_model
import numpy as np
#from numpy import linalg as lin
from collections import defaultdict as dd
import random
import configparser



class trans_model(base_model):
    def add_data(self, x, y, graph):
        
        self.x, self.y, self.graph = x, y, graph
        
        self.num_ver = max(self.graph.keys()) + 1
        
        self.y_shape=self.y.shape
        
        
        
        
    def build(self):
        
        pass
    
    def init_train(self, init_iter_label, init_iter_graph):
        """pre-training of graph embeddings.
        init_iter_label (int): # iterations for optimizing label context loss.
        init_iter_graph (int): # iterations for optimizing graph context loss.
        """
        
        gx, gy = next(self.label_generator)
        self.model_l_gy=NeuralNetUnsupervised(  self.num_ver, self.embedding_size, self.neg_samp )    
        x, y, index = next(self.inst_generator)
        self.model_l_x=NeuralNetSupervised(x.shape[1],self.y_shape[1], self.num_ver, self.use_feature, self.embedding_size, self.layer_loss,self.model_l_gy.get_embedding())
        
        
        
        criterion = nn.CrossEntropyLoss()
        criterion_loss = nn.Sigmoid()
        optimizer=torch.optim.SGD(self.model_l_gy.parameters(), lr=self.g_learning_rate)
        
        for i in range(init_iter_label):
            gx, gy = next(self.label_generator)
            #if gx.shape[0] < self.shape_init_g:continue
            l_gy=self.model_l_gy(gx)
            if self.neg_samp == 0:
                gout=torch.from_numpy(gx[:,1])
                gout=gout.long()
                g_loss =criterion(l_gy,gout)
            else:
                gy=torch.tensor(gy)
                g_loss= - torch.log(criterion_loss(torch.sum(l_gy, dim = 1) * gy)).sum()
            optimizer.zero_grad()
            g_loss.backward()
            optimizer.step()
            print ('iter label', i, g_loss)
            

        for i in range(init_iter_graph):
            gx, gy = next(self.graph_generator)
            #if gx.shape[0] < self.shape_init_g:continue
            l_gy=self.model_l_gy(gx)
            if self.neg_samp == 0:
                gout=torch.from_numpy(gx[:,1])
                gout=gout.long()
                g_loss =criterion(l_gy,gout)
            else:
                gy=torch.tensor(gy)
                g_loss= - torch.log(criterion_loss(torch.sum(l_gy, dim = 1) * gy)).sum()
            optimizer.zero_grad()
            g_loss.backward()
            optimizer.step()
            print ('iter graph', i, g_loss)

            





            
            

    def step_train(self, max_iter, iter_graph, iter_inst, iter_label):
        """a training step. Iteratively sample batches for three loss functions.
        max_iter (int): # iterations for the current training step.
        iter_graph (int): # iterations for optimizing the graph context loss.
        iter_inst (int): # iterations for optimizing the classification loss.
        iter_label (int): # iterations for optimizing the label context loss.
        """

        
        
        self.l = [self.model_l_gy, self.model_l_x]
        criterion_loss = nn.Sigmoid()
        criterion = nn.CrossEntropyLoss()
        #params = list(self.model_l_gy.parameters()) + list( self.model_l_x.parameters() )
        optimizer_gx=torch.optim.SGD( self.model_l_gy.parameters(), lr=self.g_learning_rate)
        #print(next(self.model_l_x.parameters()))
        optimizer_x=torch.optim.SGD( self.model_l_x.parameters() , lr=self.learning_rate)
                
        for _ in range(max_iter):
            for _ in range(self.comp_iter(iter_graph)):
                gx, gy = next(self.graph_generator)
                #if gx.shape[0] < self.shape_init_g:continue
                l_gy=self.model_l_gy(gx)
                if self.neg_samp == 0:
                    gout=torch.from_numpy(gx[:,1])
                    gout=gout.long()
                    loss_gy =criterion(l_gy,gout).sum()
                else:
                    gy=torch.tensor(gy)
                    loss_gy = - torch.log( criterion_loss (torch.sum(l_gy, dim = 1) * gy)).sum()
                optimizer_gx.zero_grad()
                loss_gy.backward()
                optimizer_gx.step()
            #print ('iter graph', loss_gy)

            for _ in range(self.comp_iter(iter_inst)):
                x, y, index = next(self.inst_generator)
                if self.layer_loss and self.use_feature:
                    
                    hid_sym, emd_sym, py_sym = self.model_l_x ( x, index )
                    loss = criterion( py_sym, torch.tensor(np.argmax(y, axis = 1)) ).mean()
                    
                    
                    loss += criterion( hid_sym, torch.tensor(np.argmax(y, axis = 1)) ).mean()
                    loss += criterion( emd_sym, torch.tensor(np.argmax(y, axis = 1)) ).mean()
                else:
                    py_sym = self.model_l_x ( x, index )
                    loss = criterion( py_sym,  torch.tensor(np.argmax(y, axis = 1)) ).mean()
                optimizer_x.zero_grad()
                loss.backward()
                optimizer_x.step()
                #print ('iter label', loss)
        
                
            for _ in range(self.comp_iter(iter_label)):
                gx, gy = next(self.label_generator)
                #if gx.shape[0] < self.shape_init_g:continue
                l_gy = self.model_l_gy(gx)
                if self.neg_samp == 0:
                    gout=torch.from_numpy(gx[:,1])
                    gout=gout.long()
                    loss_gy =criterion(l_gy,gout).sum()
                else:
                    gy=torch.tensor(gy)
                    loss_gy = - torch.log( criterion_loss (torch.sum(l_gy, dim = 1) * gy)).sum()
                optimizer_gx.zero_grad()
                loss_gy.backward()
                optimizer_gx.step()
            #print ('iter graph', loss_gy)

    def predict(self, tx, index = None):
        """predict the dev or test instances.
        tx (scipy.sparse.csr_matrix): feature vectors for dev instances.
        index (numpy.ndarray): indices for dev instances in the graph. By default, we use the indices from L to L + U - 1.

        returns (numpy.ndarray, #instacnes * #classes): classification probabilities for dev instances.
        """
        if index is None:
            index = np.arange(self.x.shape[0], self.x.shape[0] + tx.shape[0], dtype = np.int32)
        with torch.no_grad():
            hid_sym, emd_sym, outputs = self.model_l_x (tx, index)
            
            #predicted = torch.max(outputs, 1)[1]
            #print(predicted)
        return hid_sym, emd_sym,outputs.numpy()
    
    def gen_train_inst(self):
        """generator for batches for classification loss.
        """
        while True:
            ind = np.array(np.random.permutation(self.x.shape[0]), dtype = np.int32)
            i = 0
            while i < ind.shape[0]:
                j = min(ind.shape[0], i + self.batch_size)
                yield self.x[ind[i: j]], self.y[ind[i: j]], ind[i: j]
                i = j

    def gen_label_graph(self):
        """generator for batches for label context loss.
        """
        labels, label2inst, not_label = [], dd(list), dd(list)
        for i in range(self.x.shape[0]):
            flag = False
            for j in range(self.y.shape[1]):
                if self.y[i, j] == 1 and not flag:
                    labels.append(j)
                    label2inst[j].append(i)
                    flag = True
                elif self.y[i, j] == 0:
                    not_label[j].append(i)

        while True:
            g, gy = [], []
            for _ in range(self.g_sample_size):
                x1 = random.randint(0, self.x.shape[0] - 1)
                label = labels[x1]
                if len(label2inst) == 1: continue
                x2 = random.choice(label2inst[label])
                g.append([x1, x2])
                gy.append(1.0)
                for _ in range( self.comp_iter (self.neg_samp) ):
                    g.append([x1, random.choice(not_label[label])])
                    gy.append( - 1.0)
            yield np.array(g, dtype = np.int32), np.array(gy, dtype = np.float32)

    def gen_graph(self):
        """generator for batches for graph context loss.
        """

        

        while True:
            ind = np.random.permutation(self.num_ver)
            i = 0
            while i < ind.shape[0]:
                g, gy = [], []
                j = min(ind.shape[0], i + self.g_batch_size)
                for k in ind[i: j]:
                    if len(self.graph[k]) == 0: continue
                    path = [k]
                    for _ in range(self.path_size):
                        path.append(random.choice(self.graph[path[-1]]))
                    for l in range(len(path)):
                        for m in range(l - self.window_size, l + self.window_size + 1):
                            if m < 0 or m >= len(path): continue
                            g.append([path[l], path[m]])
                            gy.append(1.0)
                            for _ in range( self.comp_iter (self.neg_samp) ):
                                g.append([path[l], random.randint(0, self.num_ver - 1)])
                                gy.append(- 1.0)
                yield np.array(g, dtype = np.int32), np.array(gy, dtype = np.float32)
                i = j





    
        
class NeuralNetUnsupervised(nn.Module):
    def __init__(self, num_ver,embedding_size,neg_samp, **kwargs):
        '''
        input_size=size of g
        num_ver is given by base_model
        '''
        super(NeuralNetUnsupervised, self).__init__()
        
        self.num_ver = num_ver
        self.embedding_size = embedding_size
        self. neg_samp = neg_samp
        
        self.fc1 = nn.Linear(self.embedding_size, self.num_ver)
        self.nonlinearity = nn.Softmax(dim=1)
        self.embedding_l_emb_in = nn.Embedding(num_embeddings = int(self.num_ver), embedding_dim = self.embedding_size)
        if self.neg_samp > 0:
            self.embedding_l_emb_out = nn.Embedding(num_embeddings = int(self.num_ver), embedding_dim = self.embedding_size)
            
    def forward(self, g):
        l_emb_in=torch.from_numpy(g[:,0])
        l_emb_out=torch.from_numpy(g[:,1])
        '''
        g=torch.from_numpy(g)
        l_emb_in= g.narrow(1, 0, 1)
        l_emb_out= g.narrow(1, 1, 1)'''

        l_emb_in= self.embedding_l_emb_in(l_emb_in.long())
        if self.neg_samp > 0:
            l_emb_out=self.embedding_l_emb_out(l_emb_out.long())
            pgy_sym = l_emb_in * l_emb_out
        else:
            
            l_gy=self.fc1(l_emb_in)
            pgy_sym=self.nonlinearity(l_gy)
        return pgy_sym
    
    def get_embedding(self):
        return self.embedding_l_emb_in.weight
            

        
        
class NeuralNetSupervised(nn.Module):
    
    def __init__(self, x_shape_1, y_shape , num_ver, use_feature, embedding_size, layer_loss,embedding_l_emb_in,**kwargs):
        super(NeuralNetSupervised, self).__init__()
        
        self.y_shape = y_shape
        self.num_ver = num_ver
        self.use_feature = use_feature
        self.embedding_size = embedding_size
        self.layer_loss = layer_loss
        embedding_l_emb_in.requires_grad = False
        
        self.embedding_l_emb_in = nn.Embedding(num_embeddings = int(self.num_ver), embedding_dim = self.embedding_size)
        self.embedding_l_emb_in.weight = nn.Parameter(embedding_l_emb_in)
        
        
        
        #imput size of fcx = x.shape[1]
        self.fcx = nn.Linear(x_shape_1, self.y_shape )
        self.nonlinearity_x = nn.Softmax(dim=1)
        self.nonlinearity_f = nn.Softmax(dim=1)
        self.nonlinearity_y = nn.Softmax(dim=1)
        
        
        #imput size of fcf = ind.shape[0]
        
        if self.use_feature:
            self.fcf = nn.Linear(embedding_size, self.y_shape)
            self.fcy=nn.Linear( self.y_shape * 2 ,self.y_shape)
        else:
            self.fcy=nn.Linear(self.embedding_size, self.y_shape)
        
    def forward(self, x, index):
        index = torch.from_numpy(index)
        #print(self.embedding_l_emb_in.shape)
        #l_emd_f = index.long().dot(self.embedding_l_emb_in.long().t())
        l_emd_f = self.embedding_l_emb_in(index.long())
        
        #convert x to tensor
        
        x=torch.tensor(np.array(x.toarray()))
        
        l_x_hid = self.fcx(x)
        l_x_hid = self.nonlinearity_x(l_x_hid)
        if self.use_feature:
            l_emd_f = self.fcf(l_emd_f)
            l_emd_f = self.nonlinearity_f(l_emd_f)
            
            l_y = torch.cat((l_x_hid, l_emd_f), dim=1)
            
            l_y = self.fcy(l_y)
            l_y = self.nonlinearity_y(l_y)
        else:
            l_y = self.fcy(l_emd_f)
            l_y = self.nonlinearity_y(l_y)
        if self.layer_loss and self.use_feature:
            return l_x_hid, l_emd_f, l_y
        else:
            return l_y
        
 