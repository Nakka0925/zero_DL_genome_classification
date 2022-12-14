# coding: utf-8
from re import T, search
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.optimizer import *
import math

class Trainer:
    """ニューラルネットの訓練を行うクラス
    """
    def __init__(self, network, x_train, t_train, x_val, t_val, x_test, t_test,
                 epochs=10, batch_size=128,
                 optimizer='SGD', optimizer_param={'lr':0.01}, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_val = x_val
        self.t_val = t_val
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = batch_size

        # optimizer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprop':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = math.ceil(max(self.train_size / batch_size, 1))
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.test_loss_list = []
        self.train_accuracy_list = []
        self.test_accuracy_list = []

    def generator(self, x_batch, t_batch):
        images = 1 - (np.array(x_batch)) / 255
        label = 1 - (np.array(t_batch)) / 255
        images = images.reshape(1,192,192)
        label = label.reshape(1,192,192)

        yield images, label

    def train_step(self, low, high):
        #batch_mask = np.random.choice(self.train_size, self.batch_size)
        self.current_iter += 1
        x_batch = self.x_train[low:high]
        t_batch = self.t_train[low:high]
        
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        
        loss = self.network.loss(x_batch, t_batch, self.batch_size)
        print(loss)
        #self.train_loss_list.append(loss)
        #if self.verbose: print("train loss:" + str(loss))
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_val, self.t_val
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            train_loss = self.network.loss(x_train_sample, t_train_sample)
            test_loss = self.network.loss(x_test_sample, t_test_sample)
            self.train_accuracy_list.append(train_acc)
            self.test_accuracy_list.append(test_acc)
            self.train_loss_list.append(train_loss.get())
            self.test_loss_list.append(test_loss.get())

            if self.verbose:
                #end = time.time()
                print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", val acc:" + str(test_acc) + " ===")
        #self.current_iter += 1

    def train(self):
        low = 0
        high = self.batch_size
        for i in range(1, self.max_iter+1):

            self.train_step(low, high)
            low = high
            high += self.batch_size

            if (i+1) % self.iter_per_epoch == 0: 
                high = self.train_size 
            if i % self.iter_per_epoch == 0:
                low = 0
                high = self.batch_size
        
        test_acc = self.network.accuracy(self.x_test, self.t_test, search=True)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))