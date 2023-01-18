import pandas as pd
import os, sys, cv2
import yaml
import random
import numpy as np
import time


def shuffle_dataset(x, t):
    """データセットのシャッフルを行う
    Parameters
    ----------
    x : 訓練データ
    t : 教師データ
    Returns
    -------
    x, t : シャッフルを行った訓練データと教師データ
    """
    random.seed(0)
    random.shuffle(x)

    random.seed(0)
    random.shuffle(t)

    return x, t


def dataset_gain(path, creature_path, k, chk):
    """csvファイルからデータセットを作成する

    Args:
        path (str) : 作業ディレクトリのpath
        creature_path (str) : imgデータのpath
        k (int) : 分割数
        chk (int) : テストデータにするインデックス

    return:
        train_data (list) : trainのimgとlabelのlist 
        test_data (list) : testのimgとlabelのlist
    """

    train_label_list = []
    test_label_list = []
    train_img_list = []
    test_img_list = []

    for idx in range(1, k+1):
        df = pd.read_csv(path + f'cross_val/dataset/data{idx}.csv', encoding='shift-jis')
        accs = list(df['accession'])
        accs = [acc + '_0' for acc in accs]
        classes = list(df['class'])
        label_list = list(df['label'])

        if chk != idx:
            df_padding = pd.read_csv(path + f'cross_val/dataset/data{idx}_padding.csv', encoding='shift-jis')
            accs.extend(df_padding['accession'])
            classes.extend(df_padding['class'])
            label_list.extend(df_padding['label'])

            for acc, cls, label in zip(accs, classes, label_list):
                file_path = creature_path + f'img/{cls}/{acc}.png'
                train_label_list.append(label)
                train_img_list.append(file_path)
        
        else :
            for acc, cls, label in zip(accs, classes, label_list):
                file_path = creature_path + f'img/{cls}/{acc}.png' 
                test_label_list.append(label)
                test_img_list.append(file_path)


    train_img_list, train_label_list = shuffle_dataset(train_img_list, train_label_list) #TODO:paddingのcsvデータを元からシャッフルさせといた方がいいかも？


    return train_img_list, train_label_list, test_img_list, test_label_list

    """
    for i in range(192):
        for j in range(192):
            print(img_list[0][i][j], end="\t")
        print (end="\n")
    """


if __name__ == "__main__":

    with open('train_setting.yml', 'r') as yml:
        config = yaml.safe_load(yml)

    s = time.time()
    dataset_gain(config['destination'], config['creature_data_destination'], 5, chk = 1)
    print(time.time() - s)