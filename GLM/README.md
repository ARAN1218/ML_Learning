# GLM(Generalized Linear Model)
## 概要
以下の式で表される一般化線形モデルについて、Pythonで実装しました。
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon$$
※GLMについての説明を調べて書く

## リファレンス
## 重回帰分析(正規分布)
### 最小二乗法(OLS)
### 加重最小二乗法(WLS)
### 一般化最小二乗法(GLS)
### 再帰的最小二乗法(Recursive LS)


## ロジスティック回帰(二項分布)
### 二項ロジスティック回帰
- 二値分類タスクに用いられる一般化線形モデルの一種。
- リンク関数としてロジット関数、誤差構造としてベルヌーイ分布を持つ。
- パラメータの推定には最小二乗法ではなく、最尤推定法を用います。

$$\log \frac{\pi_i}{1 - \pi_i} = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + ... + \beta_n x_{in}$$
$$(E[Y_i]=\pi_i, 0<\pi_i<1)$$

<img width="480" alt="スクリーンショット 2022-12-09 0 45 58" src="https://user-images.githubusercontent.com/67265109/206492276-9759b6d7-f32d-4e1d-975d-1cc68ab1d41b.png">
<img width="221" alt="スクリーンショット 2022-12-09 0 46 29" src="https://user-images.githubusercontent.com/67265109/206492427-e377a2d3-7eef-45c0-8ecd-742fd3bbf1d7.png">


### 多項ロジスティック回帰
- 二項ロジスティック回帰の目的変数を3つ以上のカテゴリカルデータに対応させたもの。
- リンク関数として関数、誤差構造としてベルヌーイ分布を持つ。
- パラメータの推定には最小二乗法ではなく、最尤推定法を用います。

<img width="481" alt="スクリーンショット 2022-12-07 23 45 10" src="https://user-images.githubusercontent.com/67265109/206491511-3bbd9e18-e7f2-4aa0-a5df-1c3ccdc97a16.png">
<img width="433" alt="スクリーンショット 2022-12-08 23 43 43" src="https://user-images.githubusercontent.com/67265109/206491572-2d422cea-44d3-4a96-b2ab-d5257014a7a6.png">


### 順序ロジスティック分析(？)
## ポアソン回帰(ポアソン分布)
