# HuMOB2024 Rulebase

本リポジトリは、HuMOBチャレンジ2024向けの人流予測・補完・ルールベース手法・機械学習（SVR）アプローチの実装集です。

## ディレクトリ構成

- `notebook/10.common/`
  - `11_linear_interpolate.ipynb`  
    欠損値を線形補完するためのノートブック。
- `notebook/20.rulebase/`
  - `21_rulebase.ipynb`  
    ルールベース（最頻値など）による予測ノートブック。
- `notebook/30.ml/`
  - `humob-SVR_ver40_cityB.ipynb`  
    SVR（Support Vector Regression）を用いた人流予測ノートブック。
- `data/`  
  データ格納用ディレクトリ（リポジトリにはサンプルデータは含まれていません）。

## セットアップ

1. 必要なPythonパッケージをインストールしてください（例: pandas, numpy, scikit-learn, matplotlib, tqdm, geobleu など）。
2. ノートブックを実行するには、対応するデータ（例: `cityB_challengedata.csv.gz`）を`data/`ディレクトリに配置してください。

## 注意事項
- データファイルは容量が大きい場合があります。必要に応じて適切な場所に配置してください。
- 外部ライブラリ`geobleu`はノートブック内でインストールされます。