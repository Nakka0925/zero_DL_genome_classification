import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

with open('train_setting.yml', 'r') as yml:
    config = yaml.safe_load(yml)

graph_dst = Path(config['destination']) / config['data_division'] / 'graph'
graph_dst.mkdir(parents=True, exist_ok=True)

train_result = pd.read_csv(config['data_division'] + '/csv/' + config['accuracy_loss_dataname'] + '.csv', encoding = 'shift-jis')

x = np.arange(1, config['epochs']+1)

acc_dst = graph_dst / 'acc_graph'
acc_dst.mkdir(parents=True, exist_ok=True)
plt.plot(x, train_result['accuracy'], marker='o', label='train')
plt.plot(x, train_result['val_accuracy'], marker='^', label='test')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0.0, 1.0)
plt.legend(loc='lower right', fontsize=20)
plt.savefig(acc_dst / config['accuracy_graph_name'])

plt.clf()

loss_dst = graph_dst / 'loss_graph'
loss_dst.mkdir(parents=True, exist_ok=True)
plt.plot(x, train_result['loss'], marker='o', label='train')
plt.plot(x, train_result['val_loss'], marker='o', label='test')
plt.xlabel("epochs")
plt.ylabel("loss")
#plt.ylim(0, 0.25)
plt.legend(loc='upper right')
plt.savefig(loss_dst / config['loss_graph_name'])