# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 22:57:25 2015

@author:Ishay Tubi
"""

import caffe
import numpy as np


class NormlizedMSE(caffe.Layer):
    """
    Compute the normlized MSE Loss 
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match

        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
 #       self.diff = np.zeros(bottom[0].count,dtype='float32')
        # loss output is scalar
        top[0].reshape(1)
      #  print('NormlizedMSE bottom shape ',bottom[0].shape[0],bottom[0].shape[1])
                
    def forward(self,bottom, top):
        '''Mean square error of landmark regression normalized w.r.t.                                                
        inter-ocular distance                                                                                        
        '''
         # Lets assume batch size is 16 in this example remarks
        # input size is (16,10) 
        y_true = bottom[1].data # Assume second entry is the labled data
        y_pred = bottom[0].data
        
        #eye_indices = left eye x, left eye y, right eye x, right eye y, nose x, left mouth, right mouth
        #delX = y_true[:,78]-y_true[:,84] # del X size 16
        #delY = y_true[:,79]-y_true[:,85] # del y size 16
        delX = y_true[:,72]-y_true[:,90] # del X size 16
        delY = y_true[:,73]-y_true[:,91] # del y size 16


        #distXLeft = y_true[:,78]-y_true[:,54]
        #distYLeft = y_true[:,79]-y_true[:,55]
        #distXRight = y_true[:,84]-y_true[:,54]
        #distYRight = y_true[:,85]-y_true[:,55]
        #distLeft = (1e-6+(distXLeft*distXLeft + distYLeft*distYLeft)**0.5)
        #distRight = (1e-6+(distXRight*distXRight + distYRight*distYRight)**0.5)
        #dist = np.vstack((distLeft,distRight))
        #maxDist = dist.max(axis=0)
        #minDist = dist.min(axis=0)
        #ratio = (minDist/maxDist)

        self.interOc = (1e-6+(delX*delX + delY*delY)**0.5).T # Euclidain distance
        #self.interOc = self.interOc + (1e-6+(delX1*delX1 + delY1*delY1)**0.5).T # Euclidain distance
        #self.interOc = self.interOc/2.0
        #self.interOc = 1.0
        #self.interOc = (ratio*(1e-6+(delX*delX + delY*delY)**0.5)).T # Euclidain distance
        #self.interOc = ratio

        # Cannot multiply shape (16,10) by (16,1) so we transpose to (10,16) and (1,16) 
        diff = (y_pred-y_true).T # Transpose so we can divide a (16,10) array by (16,1)
        
        self.diff[...]  = (diff/self.interOc).T # We transpose back to (16,10)
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2. # Loss is scalar

        # loss = np.zeros((bottom[0].num), dtype=np.float32)
        # for k in xrange(0,68):
        #     delX = y_pred[:,k*2]-y_true[:,k*2] # del X size 16
        #     delY = y_pred[:,k*2+1]-y_true[:,k*2+1] # del y size 16
        #     loss = loss + (delX*delX + delY*delY)**0.5
        # loss = loss.T
        # top[0].data[...] = np.sum(loss/self.interOc) / bottom[0].num/68

    def backward(self, top, propagate_down, bottom):
        
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num
#            print(bottom[i].diff[...])

##################################

class EuclideanLossLayer(caffe.Layer):
    #ORIGINAL EXAMPLE
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num
            
            
            
if __name__ =="__main__":
    print '---------------------------'
    net=caffe.Net('/home/ly/workspace/Vanilla-40/ZOO/vanilla_train.prototxt', '/home/ly/workspace/Vanilla-40/ZOO/vanilaCNN.caffemodel',caffe.TRAIN)
    prediction = net.forward()['loss'][0]
    print 'lose ', prediction
     
