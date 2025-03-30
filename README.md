# 環境設定
## Conda 環境の作成
```
conda create -n test-env python=3.9.19 -y
```
僕の実行環境に合わせてpythonのバージョンを3.9.19に指定しています。他のバージョンで実行できないわけではないはずです。

## 環境を有効化
```
conda activate test-env
```
ちなみに`test-env`はてきとうに付けた名前なので、好きな名前で問題ないです◎

## 必要なパッケージのインストール
```
pip install -r requirements.txt
```

# 実行方法