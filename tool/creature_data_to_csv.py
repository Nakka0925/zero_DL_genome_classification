import pandas as pd
import os, sys, cv2
import glob
import yaml,json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold


def creature_data_gain(path):
    """DLで使用するデータをcsvにまとめる

    arge:
        path (str) : 作業ディレクトリのパス
    
    return:
        None    
    """

    df = pd.read_csv(path + 'machine-genome-classification/data/csv/learning_data.csv', encoding='shift-jis')
    df2 = pd.read_csv(path + 'machine-genome-classification/data/csv/class_sum.csv', encoding='shift-jis')

    classes = df['class']
    accs = df['accession']
    class_label = df2['class']

    with open('machine-genome-classification/data/json/label.json') as f:
        class_to_label = json.load(f)

    label = []

    for i in classes:
        if i not in list(class_label): continue 
        label.append(class_to_label[i])

    csv_dst = Path(config["destination"]) / "cross_val" / "dataset"
    csv_dst.mkdir(parents=True, exist_ok=True)

    data = {'accession' : accs, 'class' : classes, 'label' : label}

    df = pd.DataFrame(data, columns=['accession', 'class', 'label'])
    df.to_csv(csv_dst / "data_all.csv")


def creature_data_div(path,k):
    """DLで使用するデータを分割する(labelの比率は同じ)

    args:
        path (str) : 作業ディレクトリのパス
        k (int) : 分割数
    
    return:
        None
    """

    df = pd.read_csv(path + 'cross_val/dataset/data_all.csv', encoding='shift-jis') 
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    cnt = 1
    csv_dst = Path(config["destination"]) / "cross_val" / "dataset"

    for train_index, test_index in kf.split(df,df['label']): #TODO:ジェネレータでenumerate使えない？

        data = {'accession' : df['accession'][test_index], 'class' : df['class'][test_index], 'label' : df['label'][test_index]}
        split_data = pd.DataFrame(data, columns=['accession', 'class', 'label'])

        split_data.to_csv(csv_dst / f"data{cnt}.csv")
        cnt += 1


if __name__ == "__main__":

    with open('train_setting.yml', 'r') as yml:
        config = yaml.safe_load(yml)
    
    creature_data_gain(config['destination'])
    creature_data_div(config['destination'],config['fold_num'])