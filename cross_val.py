# coding: utf-8
import sys, os, time
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pandas as pd
import yaml
from common.util import shuffle_dataset
from convnet import ConvNet
from common.trainer import Trainer
from dataset import dataset_gain
from pathlib import Path

with open('train_setting.yml', 'r') as yml:
    config = yaml.safe_load(yml)


#層化抽出法を用いたK-分割交差検証
all_loss = []
all_val_loss = []
all_accuracy = []
all_val_accuracy = []
all_f1score = []

csv_dst = Path(config["destination"]) / "cross_val" / "csv"
csv_dst.mkdir(parents=True, exist_ok=True)
param_dst = Path(config["destination"]) / "cross_val" / "params"
param_dst.mkdir(parents=True, exist_ok=True)

#交差検証
######################################################################################
for idx in range(1,config['fold_num']+1):
    start = time.time()
    train_x, train_t, test_x, test_t = dataset_gain(config['destination'], config['creature_data_destination'], config['fold_num'], idx)

    train_x = train_x.reshape(-1, 1, 192, 192)
    test_x = test_x.reshape(-1, 1, 192, 192)
    #0.0 ~ 1.0 に正規化　※1から引いているのは画素値の値を反転させるため
    train_x = 1 - train_x / 255
    test_x = 1 - test_x / 255
    #shuffle
    train_x, train_t =  shuffle_dataset(train_x, train_t)
    
    network = ConvNet(k=idx)

    trainer = Trainer(network, train_x, train_t, test_x, test_t,
                epochs=config['epochs'], batch_size = config["batch_size"],
                optimizer='Adam', optimizer_param={'lr': 0.001}
                )

    trainer.train()

    accuracy = trainer.train_accuracy_list
    val_accuracy = trainer.test_accuracy_list
    loss = trainer.train_loss_list
    val_loss = trainer.test_loss_list

    data = {'accuracy' : accuracy, 'val_accuracy' : val_accuracy, 'loss' : loss, 'val_loss' : val_loss}
    df = pd.DataFrame(data, columns=['accuracy', 'val_accuracy', 'loss', 'val_loss'])
    df.index = ['epoch ' + str(n) for n in range(1, config['epochs']+1)]
    df.to_csv(csv_dst / (config['accuracy_loss_dataname'] + str(idx) + '.csv'))

    all_accuracy.append(accuracy)
    all_val_accuracy.append(val_accuracy)
    all_loss.append(loss)
    all_val_loss.append(val_loss)
    all_f1score.append(network.f1score)
    
    network.save_params(param_dst / (config['params_name'] + str(idx) + '.pkl'))
    print("time" + str(time.time() - start))
    print("Saved Network Parameters!")
######################################################################################


#accuracy, lossの平均
ave_accuracy = np.mean(all_accuracy, axis = 0)
ave_val_accuracy = np.mean(all_val_accuracy, axis = 0)
ave_loss = np.mean(all_loss, axis = 0)
ave_val_loss = np.mean(all_val_loss, axis = 0)

data = {'accuracy' : ave_accuracy, 'val_accuracy' : ave_val_accuracy, 'loss' : ave_loss, 'val_loss' : ave_val_loss}
df = pd.DataFrame(data, columns=['accuracy', 'val_accuracy', 'loss', 'val_loss'])
df.index = ['epoch ' + str(n) for n in range(1, config['epochs']+1)]
df.to_csv(csv_dst / (config['accuracy_loss_dataname'] + "_all.csv"))

#f1score
data = {'f1score' : all_f1score}
df = pd.DataFrame(data, columns=['f1score'])
df.index = ['k' + str(n) for n in range(1, config['fold_num']+1)]
df.to_csv(csv_dst / (config['f1score_dataname'] + ".csv"))