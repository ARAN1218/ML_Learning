class SHAP:
    """SHapley Additive exPlanations
    
    Args:
        estimator: 学習済みモデル(Any)
        X: SHAPの計算に使う特徴量(pd.DataFrame)
        var_names: 特徴量の名前(list[str])
    """
        
    def __init__(self, estimator, X) -> None:
        self.estimator = estimator
        self.X = X
        self.var_names = list(self.X.columns)
        
        # ベースラインとしての平均的な予測値
        self.baseline = self.estimator.predict(self.X).mean()

        # 特徴量の総数
        self.J = self.X.shape[1]

        # あり得るすべての特徴量の組み合わせ
        self.subsets = [
            s
            for j in range(self.J + 1)
            for s in combinations(range(self.J), j)
        ]

    def _get_expected_value(self, subset) -> np.ndarray:
        """特徴量の組み合わせを指定するとその特徴量が場合の予測値を計算

        Args:
            subset: 特徴量の組み合わせ(tuple[int, ...])
        """
        
        _X = self.X.copy()  # 元のデータが上書きされないように

        # 特徴量がある場合は上書き。なければそのまま。
        if len(subset) > 0 :
            # 元がtupleなのでリストにしないとインデックスとして使えない
            _s = list(subset)
            _X.iloc[:, _s] = list(_X.iloc[self.i, _s])

        return self.estimator.predict(_X).mean()

    def _calc_weighted_marginal_contribution(self, j, s_union_j) -> float:
        """限界貢献度x組み合わせ出現回数を求める

        Args:
            j: 限界貢献度を計算したい特徴量のインデックス(int)
            s_union_j: jを含む特徴量の組み合わせ(tuple[int, ...])
        """
        
        # 特徴量jがない場合の組み合わせ
        s = tuple(set(s_union_j) - set([j]))

        # 組み合わせの数
        S = len(s)

        # 組み合わせの出現回数
        # ここでfactorial(self.J)で割ってしまうと丸め誤差が出てるので、あとで割る
        weight = factorial(S) * factorial(self.J - S - 1)

        # 限界貢献度
        marginal_contribution = (
            self.expected_values[s_union_j] - self.expected_values[s]
        )

        return weight * marginal_contribution

    def shap(self, id_to_compute) -> None:
        """SHAP値を求める

        Args:
            id_to_compute: SHAPを計算したいインスタンス(int)
        """

        # SHAPを計算したいインスタンス
        self.i = id_to_compute

        # すべての組み合わせに対して予測値を計算
        # 先に計算しておくことで同じ予測を繰り返さずに済む
        self.expected_values = {
            s: self._get_expected_value(s) for s in self.subsets
        }

        # ひとつひとつの特徴量に対するSHAP値を計算
        shap_values = np.zeros(self.J)
        for j in range(self.J):
            # 限界貢献度の加重平均を求める
            # 特徴量jが含まれる組み合わせを全部もってきて
            # 特徴量jがない場合の予測値との差分を見る
            shap_values[j] = np.sum([
                self._calc_weighted_marginal_contribution(j, s_union_j)
                for s_union_j in self.subsets
                if j in s_union_j
            ]) / factorial(self.J)
        
        # データフレームとしてまとめる
        self.df_shap = pd.DataFrame(
            data={
                "var_name": self.var_names,
                "feature_value": self.X.iloc[id_to_compute],
                "shap_value": shap_values,
            }
        )

    def plot(self) -> None:
        """SHAPを可視化"""
        
        # 下のデータフレームを書き換えないようコピー
        df = self.df_shap.copy()
        
        # グラフ用のラベルを作成
        df['label'] = [
            f"{x} = {y:.2f}" for x, y in zip(df.var_name, df.feature_value)
        ]
        
        # SHAP値が高い順に並べ替え
        df = df.sort_values("shap_value").reset_index(drop=True)
        
        # 全特徴量の値がときの予測値
        predicted_value = self.expected_values[self.subsets[-1]]
        
        # 棒グラフを可視化
        fig, ax = plt.subplots()
        ax.barh(df.label, df.shap_value)
        ax.set(xlabel="SHAP値", ylabel=None)
        fig.suptitle(f"SHAP値 \n(Baseline: {self.baseline:.2f}, Prediction: {predicted_value:.2f}, Difference: {predicted_value - self.baseline:.2f})")

        fig.show()

        
# テスト------------------------------------------------------------------------------------------------------------------
# Random Forestで予測モデルを構築
rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42).fit(X_train, y_train)
# 予測精度の評価
display(regression_metrics(rf, X_test, y_test))

# SHAPのインスタンスを作成
shap = SHAP(rf, X_test)
# インスタンス1に対してSHAP値を計算
shap.shap(id_to_compute=1)
# SHAP値を出力
display(shap.df_shap)
# SHAP値を可視化
shap.plot()
