# ミトコンドリアゲノムのグラフ表示画像を入力とする機械学習ネットワークの構築

## イントロダクション　

従来のゲノム解析ではアライメントと呼ばれる方法が用いられているが, これは比較する配列の長さが互いに $N$ の場合, $O(N^2)$ の時間計算量がかかり, ゲノム配列のような莫大な長さの配列の類似性を決定するには多くの処理時間を要する欠点がある. そのため本研究室ではアライメントに依らない方法として, ミトコンドリアゲノム配列に現れる 4 種の塩基 A, T, G, C に対してベクトルを割り振ることでゲノム配列をグラフ画像化し, 作成された画像から機械学習を用いることにより, 未知の生物がどの生物分類に属しているかを判定する方法について研究されてきた.\
先行研究では機械学習をする際に, 機械学習専用のフレームワークを用いて行っていたが, 本研究では新たな試みとして機械学習のネットワークをゼロから作成し分類精度の向上を目指す

## 前準備
1. リポジトリをクローンする
```console
$ git clone https://github.com/Nakka0925/zero_DL_genome_classification.git
```
2. 画像データのダウンロード
```console
$ ./data_download.sh  
$ unzip img_data.zip  #展開
```
## 使い方
1. pipenvを使って仮想環境を再現する
python3.8系を指定しているので別途pyenvなどで指定する
```console
$ cd zero_DL_genome_classification
$ pipenv install 
```
2. 初期設定をする
```console
$ pipenv run python tool/initialize.py
```
`train_setting.yml`が生成されるので各自でパラメータを設定する

3. 学習の実行
```console
$ pipenv run learn
```
4. グラフの生成
```console
$ pipenv run graph
```