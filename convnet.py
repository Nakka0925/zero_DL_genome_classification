# coding: utf-8
from re import T
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
import numpy as np
from common.layers import *
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import yaml,json

class ConvNet:
    """ネットワーク構成は下記の通り
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
    """
    def __init__(self, input_dim=(1, 192, 192),
                 conv_param_1 = {'filter_num':8, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2 = {'filter_num':8, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_3 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_4 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_5 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_6 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size=64, output_size=49, k = None):
        # 重みの初期化===========
        # 各層のニューロンひとつあたりが、前層のニューロンといくつのつながりがあるか（TODO:自動で計算する）
        pre_node_nums = np.array([1*3*3, 8*3*3, 8*3*3, 16*3*3, 16*3*3, 32*3*3, 32*24*24, hidden_size])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLUを使う場合に推奨される初期値
        self.k = k
        self.f1score = 0
        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        self.params['W7'] = weight_init_scales[6] * np.random.randn(32*24*24, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = weight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        # レイヤの生成===========
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'], 
                           conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W2'], self.params['b2'], 
                           conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W3'], self.params['b3'], 
                           conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                           conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                           conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W6'], self.params['b6'],
                           conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Dropout(0.3))
        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.3))
        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        
        self.last_layer = SoftmaxWithLoss()

    def create_heatmap(self, t, pri):
        label_sum = {}
        for label in t:
            label_sum.setdefault(label,0)
            label_sum[label] += 1
        label_sum =  dict(sorted(label_sum.items()))
        cm = confusion_matrix(t, pri)
        self.f1score = f1_score(t, pri, average='macro')
        cm = cm.astype(np.float16)

        for idx in range(len(label_sum)):
            cm[idx] = cm[idx] / label_sum[idx]
        
        cm = np.round(cm, decimals=2)
    
        with open('train_setting.yml', 'r') as yml:
            config = yaml.safe_load(yml)

        graph_dst = Path(config['destination']) / config['data_division'] / 'graph'
        graph_dst.mkdir(parents=True, exist_ok=True)
        graph_dst = graph_dst / 'confusion_matrix'
        graph_dst.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(config['creature_data_destination'] + 'csv/class_sum.csv', encoding='shift-jis')
        fig, axes = plt.subplots(figsize=(22,23))
        cm = pd.DataFrame(data=cm, index=df['class'], columns=df['class'])

        sns.heatmap(cm, square=True, cbar=True, annot=True, cmap = 'Blues', vmax=1, vmin=0)
        plt.xlabel("Pre", fontsize=15, rotation=0)
        plt.xticks(fontsize=15)
        plt.ylabel("Ans", fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig(graph_dst / (config['heat_map_name'] + str(self.k) + '.png')) 
        plt.close()


    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t,batch_size = 128):
        """
        y = self.predict(x, train_flg=True)
        #print (y)
        return self.last_layer.forward(y, t)
        """

        tmp = cp.asarray(0, dtype=cp.float32)
        n = int(x.shape[0] / batch_size)
        for i in range(n):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=True)
            tmp += self.last_layer.forward(y,tt)

        if x.shape[0] % batch_size != 0:
            tx = x[n*batch_size:]
            tt = t[n*batch_size:]
            y = self.predict(tx, train_flg=True)
            tmp += self.last_layer.forward(y,tt)
        
        return tmp / x.shape[0]

    def accuracy(self, x, t, batch_size=128, search=False):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        pri = np.empty(0, dtype=np.float16)

        acc = 0.0
        n = int(x.shape[0] / batch_size)
        for i in range(n):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = y.get()
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
            if search:
                pri = np.append(pri, y)
        
        if x.shape[0] % batch_size != 0:
            tx = x[n*batch_size:]
            tt = t[n*batch_size:]
            y = self.predict(tx, train_flg=False)
            y = y.get()
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
            if search:
                pri = np.append(pri, y)

        if search:
            self.create_heatmap(t,pri)
            
        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 16, 19)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW.get()
            grads['b' + str(i+1)] = self.layers[layer_idx].db.get()

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]