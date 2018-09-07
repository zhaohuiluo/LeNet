import numpy as np
import time
import sys
import random
import struct

class convLayer(object):
    def __init__(self,c_in,c_out, k, name,lr, d= 1, p= 1):
        self.w = np.random.randn(c_out,c_in, k, k)
        self.b = np.random.randn(c_out)		
        self.k=k
        self.c_out=c_out
        self.lr = lr
        self.d = d
        self.p = p
        self.name=name

    def forward(self,data_in):

        self.data_in=data_in
        N,C, H, W = data_in.shape
        d, p = self.d, self.p
        h_out = int(1 + (H + 2 * p - self.k) /d)
        w_out = int(1 + (W + 2 * p - self.k) /d)
        data_out = np.zeros((N , self.c_out , h_out, w_out))
        data_pad = np.pad(data_in, ((0,), (0,), (p,), (p,)), mode='constant', constant_values=0)

        for i in range(h_out):
            for j in range(w_out):
                data_block = data_pad[:, :, i*d:i*d+self.k, j*d:j*d+self.k]
                for t in range(self.c_out):
                    data_out[:, t , i, j] = np.sum(data_block * self.w[t, :, :, :], axis=(1,2,3))+self.b[t]
        #print(self.name,data_out.shape)
        return data_out

    def backward(self,diff_top):
        N, C, H, W = self.data_in.shape
        d, p = self.d, self.p
        k=self.k
        h_out = int(1 + (H + 2 * p - k) / d)
        w_out = int(1 + (W + 2 * p - k) / d)

        data_pad = np.pad(self.data_in, ((0,), (0,), (p,), (p,)), mode='constant', constant_values=0)
        diff_out = np.zeros_like(self.data_in,dtype=np.float64)
        diff_pad = np.zeros_like(data_pad,dtype=np.float64)
        diff_w = np.zeros_like(self.w)
        diff_b = np.zeros_like(self.b)

        diff_b = np.sum(diff_top, axis=(0, 2, 3))
		
        for i in range(h_out):
            for j in range(w_out):
                data_block = data_pad[:, :, i *d:i *d +k, j * d:j *d + k]
                for t in range(self.c_out): 
                    diff_w[t, :, :, :] += np.sum(data_block * (diff_top[:, t, i, j])[:, None, None, None], axis=0)
                for idx in range(N):  
                    diff_pad[idx, :, i *d:i *d+k, j*d:j*d+k] += np.sum(self.w[:, :, :, :] *(diff_top[idx, :, i,j])[:, None, None,None], axis=0)
        diff_out[:,:,:,:] = diff_pad[:, :, p:-p, p:-p]
        self.w -= self.lr * diff_w 
        self.b -= self.lr * diff_b
        #print(self.name,diff_out.shape)
        return diff_out


class fcLayer:
    def __init__(self, c_in, c_out,name,lr):
        self.c_in = c_in
        self.c_out = c_out
        self.w =np.random.randn(c_in, c_out)*1e-4
        self.b =np.zeros(self.c_out)*1e-4
        self.lr = lr
        self.name=name

    def forward(self, data_in):
        self.data_in = data_in
        data_out = data_in.dot(self.w) + self.b
        #print(data_out.shape)
        return data_out
		
    def backward(self, diff_top):
        diff_out=np.zeros_like(self.data_in )
        diff_out = diff_top.dot(self.w.T)
        self.w -= self.lr * self.data_in.T.dot(diff_top)
        self.b -= self.lr * np.sum(diff_top, axis=0)
        #print(self.name,diff_out.shape)
        return diff_out


class reluLayer:
    def __init__(self,name):
        self.name=name

    def forward(self, data_in):

        data_out = data_in
        data_out[data_out<0]=0
        self.data_out=data_out
        #print(self.name,data_out.shape)
        return data_out

    def backward(self,diff_top):
        diff_out=(self.data_out > 0) * diff_top
        #print(self.name,diff_out.shape)
        return diff_out


class maxPoolingLayer:
    def __init__(self, k,name,d=1):
        self.k = k
        self.d = d
        self.name=name

    def forward(self, data_in):
        self.data_in = data_in
        n, c, h, w = data_in.shape
        k,d=self.k,self.d
        h_out = int((h - k) /d + 1)
        w_out = int((w - k) /d + 1)
        data_out = np.zeros((n, c, h_out, w_out))
        for i in range(h_out):
            for j in range(w_out):
                data_block = data_in[:, :, i * d: i * d + k, j * d: j * d + k]
                data_out[:, :, i, j] = np.max(data_block, axis=(2, 3))
        #print(self.name,data_out.shape)
        return data_out

    def backward(self, diff_top):
        n, c, h, w = self.data_in.shape
        k,d=self.k,self.d
        h_out = int((h - k) /d + 1)
        w_out = int((w - k) /d + 1)
        diff_out = np.zeros_like(self.data_in)

        for i in range(h_out):
            for j in range(w_out):
                data_block = self.data_in[:, :, i * d: i *d + k, j *d: j *d + k]
                block_max = np.max(data_block, axis=(2, 3))
                diff_block = (data_block == (block_max)[:, :, None, None])
                diff_out[:, :, i *d: i *d + k, j * d: j * d + k] += diff_block * (diff_top[:, :, i,j])[:, :, None, None]
        #print(self.name,diff_out.shape)
        return diff_out


class flattenLayer:
    def __init__(self,name):
        self.name=name

    def forward(self, data_in):
        self.data_in=data_in
        n, c, h, w = data_in.shape
        data_out=np.reshape(data_in,(n, c*h*w))
        #print(self.name,data_out.shape)
        return data_out

    def backward(self, diff_top):
        n,c,h,w= self.data_in.shape
        diff_out=np.reshape(diff_top,(n,c,h,w))
        #print(self.name,diff_out.shape)
        return diff_out


class softmaxLayer:
    def __init__(self,name):
        self.name=name

    def forward(self,data_in):
        data_in = data_in - np.max(data_in, axis=1).reshape(-1, 1)
        self.data_out = np.exp(data_in) / np.sum(np.exp(data_in), axis=1).reshape(-1, 1)
        #print(self.name,self.data_out.shape)
        return self.data_out

    def backward(self, diff_top):
        N = diff_top.shape[0]
        diff_out= self.data_out.copy()
        diff_out[range(N), list(diff_top)] -= 1
        diff_out /= N
        #print(self.name,diff_out.shape)
        return diff_out

class Net:
    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def train(self, trainData, trainLabel, validData, validLabel, batch_size, iteration):
        train_num = trainData.shape[0]
        strainData = trainData
        strainLabel = trainLabel
        for iter in range(iteration):
            index = np.random.choice([ i for i in range(train_num)], train_num)
            # index = [i for i in range(train_num)]
            trainData = strainData[index]
            trainLabel = strainLabel[index]

            if iter > 100:
                lay_num = len(self.layers)
                for i in range(lay_num):
                   self.layers[i].lr *= (0.001 ** ( iter - 100 ) / 100)

            print(str(time.clock()) + '  iter=' + str(iter))
            for batch_iter in range(0, train_num, batch_size):
                if batch_iter + batch_size < train_num:
                    loss = self.train_inner(trainData[batch_iter: batch_iter + batch_size],
                                     trainLabel[batch_iter: batch_iter + batch_size])
                else:
                    loss = self.train_inner(trainData[batch_iter: train_num],
                                     trainLabel[batch_iter: train_num])
                print(str(batch_iter) + '/' + str(train_num) + '   loss : ' + str(loss))
            print(str(time.clock()) + "  eval=" + str(self.test(trainData, trainLabel)))
            print(str(time.clock()) + "  eval=" + str(self.test(validData, validLabel)))

    def train_inner(self, data_in, label):
        l= len(self.layers)
        for i in range(l):
            data_out = self.layers[i].forward(data_in)
            data_in = data_out
        num= data_out.shape[0]
        loss = -np.sum(np.log(data_out[range(num), list(label)]))
        loss /= num
        diff_in = label
        for i in range(0, l):
            diff_out = self.layers[l - i - 1].backward(diff_in)
            diff_in = diff_out
        return loss

    def test(self, data_in, label):
        l= len(self.layers)
        for i in range(l):
            data_out = self.layers[i].forward(data_in)
            data_in = data_out
        y = np.argmax(data_in, axis=1)
        return np.sum(y == label) / float(y.shape[0])
        
def loadImageSet(filename):
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)
    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    #print(head)
    # [60000]*28*28
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'  # like '>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    #imgs = np.reshape(imgs, [imgNum, -1])
    imgs=np.reshape(imgs,(imgNum,1,width,height))
    #print(imgs.shape)
    return imgs

def loadLabelSet(filename):
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)
    imgNum = head[1]

    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])

    ret = labels
    return ret

rate = 1e-5
net = Net()
net.addLayer(convLayer(1, 32, 5,'conv1',rate))
net.addLayer(maxPoolingLayer(3,'pooling1',2))
net.addLayer(reluLayer('relu1'))

net.addLayer(convLayer(32, 16, 3,'conv2',rate))
net.addLayer(maxPoolingLayer(3,'pooling2',2))
net.addLayer(reluLayer('relu2'))

net.addLayer(flattenLayer('flat'))
net.addLayer(fcLayer(5 * 5 * 16, 100,'fc1',rate))
net.addLayer(reluLayer('relu1'))
net.addLayer(fcLayer(100, 10,'fc2',rate))
net.addLayer(softmaxLayer('softloss'))

path='/home/luo/'

train_feature = loadImageSet(path+'data/MNIST_data/train-images.idx3-ubyte')
train_label = loadLabelSet(path+'data/MNIST_data/train-labels.idx1-ubyte')
valid_feature = loadImageSet(path+'data/MNIST_data/t10k-images.idx3-ubyte')
valid_label = loadLabelSet(path+'data/MNIST_data/t10k-labels.idx1-ubyte')

N = 1000
M = 100
net.train(train_feature, train_label , valid_feature[0:100],valid_label[0:100],10,200)

