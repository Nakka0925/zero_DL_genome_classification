# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
from common.util import shuffle_dataset
from dataset import raw_data_gain, padding_data_gain
from sklearn.model_selection import train_test_split
from convnet import ConvNet
from pathlib import Path
import yaml

with open('train_setting.yml', 'r') as yml:
    config = yaml.safe_load(yml)

params_dst = Path(config['destination']) / config['data_division'] / 'params' / config['params_dataname']

file_list, label_list = raw_data_gain(config['img_destination'])
train_x, test_x, train_y, test_y = train_test_split(file_list, label_list, test_size=0.2,
                                                    random_state=1, stratify = label_list)
    
"""
file_list_padding, label_list_padding = padding_data_gain(contig['destination'])
padding_train_x, padding_test_x, padding_train_y, padding_test_y = train_test_split(file_list_padding, label_list_padding, test_size=0.2,
                                                    stratify = label_list_padding)

train_x.extend(padding_train_x)
train_y.extend(padding_train_y)
"""
#train_x = np.array(train_x, dtype=np.float16)
test_x = np.array(test_x, dtype=np.float16)
#train_y = np.array(train_y, dtype=np.int8)
test_y = np.array(test_y, dtype=np.int8)

#train_x = train_x.reshape(-1, 1, 192, 192)
test_x = test_x.reshape(-1, 1, 192, 192)
#0.0 ~ 1.0 に正規化　※1から引いているのは画素値の値を反転させるため
#train_x = 1 - train_x / 255
test_x = 1 - test_x / 255
#shuffle
#train_x, train_y =  shuffle_dataset(train_x, train_y)

network = ConvNet()
network.load_params(params_dst)

print(network.accuracy(test_x, test_y, search=True))