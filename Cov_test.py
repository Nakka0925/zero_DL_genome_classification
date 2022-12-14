# coding: utf-8
import sys, os

from common.util import shuffle_dataset
from convnet import ConvNet
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from dataset import raw_data_gain, padding_data_gain
from simple_convnet import SimpleConvNet
from common.trainer import Trainer
import pandas as pd
import yaml

with open('train_setting.yml', 'r') as yml:
    contig = yaml.safe_load(yml)

file_list, label_list = raw_data_gain(contig['img_destination'])
train_x, test_x, train_y, test_y = train_test_split(file_list, label_list, test_size=0.2,
                                                    random_state=1, stratify = label_list)

file_list_padding, label_list_padding = padding_data_gain(contig['img_destination'])
padding_train_x, padding_test_x, padding_train_y, padding_test_y = train_test_split(file_list_padding, label_list_padding, test_size=0.2,
                                                    random_state=1, stratify = label_list_padding)

train_x.extend(padding_train_x)
train_y.extend(padding_train_y)


train_x = np.array(train_x, dtype=np.float16)
test_x = np.array(test_x, dtype=np.float16)
train_y = np.array(train_y, dtype=np.int8)
test_y = np.array(test_y, dtype=np.int8)

train_x = train_x.reshape(-1, 1, 192, 192)
test_x = test_x.reshape(-1, 1, 192, 192)
#0.0 ~ 1.0 に正規化　※1から引いているのは画素値の値を反転させるため
train_x = 1 - train_x / 255
test_x = 1 - test_x / 255
#shuffle
train_x, train_y =  shuffle_dataset(train_x, train_y)

max_epochs = 30

"""
#層化抽出法を用いたK-分割交差検証
splits = 5
kf = StratifiedKFold(n_splits=splits, shuffle=True)

all_loss = []
all_val_loss = []
all_acc = []
all_val_acc = []

#交差検証
for train_index, val_index in kf.split(file_list, label_list):              
    
    train_data  = file_list[train_index]
    train_label = label_list[train_index]
    val_data    = file_list[val_index]
    val_label   = label_list[val_index]

    network = SimpleConvNet(input_dim=(1,192,192), 
                        conv_param = {'filter_num': 8, 'filter_size': 3, 'pad': 0, 'stride': 1},
                        hidden_size=64, output_size=3, weight_init_std='Relu', use_dropout = use_dropout,
                        dropout_ration = dropout_ratio
                        )
    
    network = DeepConvNet()

    trainer = Trainer(network, train_data, train_label, val_data, val_label, test_x, test_y,
                    epochs=max_epochs, batch_size = 128,
                    optimizer='Adam', optimizer_param={'lr': 0.001}
                    )

    trainer.train()

    acc = trainer.train_acc_list
    val_acc = trainer.test_acc_list
    loss = trainer.train_loss_list
    val_loss = trainer.test_loss_list

    all_acc.append(acc)
    all_val_acc.append(val_acc)
    all_loss.append(loss)
    all_val_loss.append(val_loss)

#accuracy, loss平均の平均
ave_acc = np.mean(all_acc, axis = 0)
ave_val_acc = np.mean(all_val_acc, axis = 0)
ave_loss = np.mean(all_loss, axis = 0)
ave_val_loss = np.mean(all_val_loss, axis = 0)
"""


#検証データなし
train_data  = train_x
train_label = train_y
val_data    = test_x
val_label   = test_y
"""
network = SimpleConvNet(input_dim=(1,192,192), 
                        conv_param = {'filter_num': 8, 'filter_size': 3, 'pad': 1, 'stride': 1},
                        hidden_size=128, output_size=49, weight_init_std='Relu', use_dropout = use_dropout,
                        dropout_ration = dropout_ratio
                        )
#"""
network = ConvNet()

trainer = Trainer(network, train_data, train_label, val_data, val_label, test_x, test_y,
                epochs=max_epochs, batch_size = 128,
                optimizer='Adam', optimizer_param={'lr': 0.001}
                )

trainer.train()

acc = trainer.train_acc_list
val_acc = trainer.test_acc_list
loss = trainer.train_loss_list
val_loss = trainer.test_loss_list

data = {'acc' : acc, 'val_acc' : val_acc, 'loss' : loss, 'val_loss' : val_loss}
df = pd.DataFrame(data, columns=['acc', 'val_acc', 'loss', 'val_loss'])
df.index = ['epoch ' + str(n) for n in range(1, max_epochs+1)]
df.to_csv(contig['accuracy_loss_dataname'])

# パラメータの保存
network.save_params(contig['params_dataname'])
print("Saved Network Parameters!")