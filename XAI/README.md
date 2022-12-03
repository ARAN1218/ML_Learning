# XAI(eXplainable Artificial Intelligence)

## 概要
ブラックボックスな機械学習を「解釈する」技術を実装しました。

## リファレンス
### PFI(Permutation Feature Importance)
- ある特徴量をシャッフルして予測した際の誤差の増加量・増加率を特徴量の重要度として定義したもの。
- 「**本当に重要な特徴量ならば、適当にシャッフルされた場合に予測値に大きな影響を与えるだろう**」という考えから成り立つ。
- どんな機械学習モデルに対しても適用可能である。(モデル非依存)
- モデルの学習自体を繰り返さないため、比較的計算負荷が低い。
- この値を見ることで、「**ある機械学習モデルにおいてどの特徴量を予測に大きく活用しているか**」が分かる。
  - あくまでモデルの振る舞い(モデルの予測値と特徴量との関係)を見ているのであって、因果関係(目的変数自体と特徴量との関係)について言及できるものではないことに注意が必要である。
  - モデルの振る舞いとして強い相関の示唆が出たとしても、真に重要な特徴量がモデルに組み込まれていない等の疑似相関に注意が必要である。
- 特徴量間で相関性がある場合は、重要度の食い合いが発生して計算がおかしくなってしまう。
  - 該当する特徴量を一つのグループにまとめて値をシャッフルする等の対策を取る必要がある。
- PFIの計算に用いるデータは、モデルがよく汎化されている場合は学習・テストデータのどちらを使っても問題ない。

### 求め方
1. K個の特徴量 $X_0,X_1...X_k$ を用いて学習させた任意の機械学習モデル $f$ を用意し、テストデータでベースラインとなる予測値を出力する。
2. テストデータのある特徴量 $X_0$ の値のみを無作為にシャッフルし、このデータで同モデルを用いて予測値を出す。
3. 1と2の出力を比較し、予測誤差の増加量・増加率を計算し、これを特徴量 $X_0$ の特徴量重要度とする。
4. 2と3の工程を何回か行い、特徴量重要度の平均を取ると良い。
5. 残りの特徴量 $X_1,X_2...X_k$ についても2~4の工程を行い、特徴量重要度が出揃った時点で各特徴量の重要度を比較分析する。

![Unknown](https://user-images.githubusercontent.com/67265109/205437514-79fd972f-9b3a-4e11-98fb-0b6c6610bf2d.png)


### LOCOFI(Leave One Covariate Out Feature Importance)
- 




### GPFI(Grouped PFI)




### PDP(Partial Dependence)




### ICE(Individual Conditional Expectation)




### PDP&ICE




### CPD(Conditional Partial Dependence)





### SHAP(SHapley Additive exPlanations)





## 参考文献
- 機械学習を解釈する技術　森下光之助　技術評論社(2021-08-17)
- [Interpretable Machine Learning](https://hacarus.github.io/interpretable-ml-book-ja/)　Christoph Molnar　(2021-05-31)


![Unknown](https://user-images.githubusercontent.com/67265109/202885565-60e3bc42-248b-4bec-9a4e-436c74c439d9.jpeg)
![Unknown-1](https://user-images.githubusercontent.com/67265109/202885613-a747b1ea-a04d-481c-b023-00605665fe40.jpeg)

