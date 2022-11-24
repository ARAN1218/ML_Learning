class GPFI(PFI):
    """Grouped Permutation Feature Importance (GPFI)"""

    def _permutation_metrics(self, var_names_to_permute) -> float:
        """ある特徴量群の値をシャッフルしたときの予測精度を求める

        Args:
            var_names_to_permute: シャッフルする特徴量群の名前(list[str])
        """

        # シャッフルする際に、元の特徴量が上書きされないよう用にコピーしておく
        X_permuted = self.X.copy()

        # 特徴量群をまとめてシャッフルして予測
        X_permuted[var_names_to_permute] = X_permuted[var_names_to_permute].sample(frac=1).reset_index(drop=True)
        y_pred = self.estimator.predict(X_permuted)

        return mean_squared_error(self.y, y_pred, squared=False)

    def permutation_feature_importance(self, var_groups=None, n_shuffle=10) -> None:
        """GPFIを求める

        Args:
            var_groups:
                グループ化された特徴量名のリスト。例：[['X0', 'X1'], ['X2']]
                Noneを指定すれば通常のPFIが計算される(list[list[str]] | None)
            n_shuffle:
                シャッフルの回数。多いほど値が安定する。デフォルトは10回(int)
        """

        # グループが指定されなかった場合は1つの特徴量を1グループとする。PFIと同じ。
        if var_groups is None:
            var_groups = [[j] for j in self.var_names]

        # グループごとに重要度を計算
        # R回シャッフルを繰り返して値を安定させている
        metrics_permuted = [
            np.mean(
                [self._permutation_metrics(fn) for r in range(n_shuffle)]
            )
            for fn in var_groups
        ]

        # データフレームとしてまとめる
        # シャッフルでどのくらい予測精度が落ちるかは、差と比率の2種類を用意する
        df_feature_importance = pd.DataFrame(
            data={
                "var_name": ["+".join(j) for j in var_groups],
                "baseline": self.baseline,
                "permutation": metrics_permuted,
                "difference": metrics_permuted - self.baseline,
                "ratio": metrics_permuted / self.baseline,
            }
        )

        self.feature_importance = df_feature_importance.sort_values(
            "permutation", ascending=False
        )

        
# テスト------------------------------------------------------------------------------------------------------------------
# 特徴量X2と全く同じ特徴量を追加
X_train2 = pd.concat([X_train, X_train['X2']], axis=1).set_axis(["X0", "X1", "X2", "X3"], axis=1)
# 特徴量X2と全く同じ特徴量を追加した新しいデータからRandom Forestの予測モデルを構築
rf = RandomForestRegressor(n_jobs=-1, random_state=42).fit(X_train2, y_train)
# テストデータにも同様に特徴量X2とまったく同じ値をとる特徴量X3を作る。
X_test2 = pd.concat([X_test, X_test['X2']], axis=1).set_axis(["X0", "X1", "X2", "X3"], axis=1)

# var_groupsを指定しなければ通常のPFIが計算される
gpfi = GPFI(rf, X_test2, y_test, var_names=["X0", "X1", "X2", "X3"])
gpfi.permutation_feature_importance()
gpfi.plot()

# X2とX3はまとめてシャッフル。X0とX1は個別にシャッフル
gpfi.permutation_feature_importance(var_groups=[["X0"], ["X1"], ["X2", "X3"]])
# GPFIを可視化
gpfi.plot()
