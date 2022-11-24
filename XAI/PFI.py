class PFI:
    """Permutation Feature Importance (PFI)
     
    Args:
        estimator: 全特徴量を用いた学習済みモデル(Any)
        X: 特徴量(pd.DataFrame)
        y: 目的変数(pd.DataFrame)
        var_names: 特徴量の名前(list[str])
    """

    # クラスインスタンスを定義
    def __init__(self, estimator, X, y, var_names):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.var_names = var_names
        
        # シャッフルなしの場合の予測精度
        # mean_squared_error()はsquared=TrueならMSE、squared=FalseならRMSE
        self.baseline = mean_squared_error(
            self.y, self.estimator.predict(self.X), squared=False
        )

    def _permutation_metrics(self, feature_to_permute: str) -> float:
        """ある特徴量の値をシャッフルしたときの予測精度を求める

        Args:
            feature_to_permute: シャッフルする特徴量の名前
        """

        # シャッフルする際に、元の特徴量が上書きされないよう用にコピーしておく
        X_permuted = self.X.copy()

        # 特徴量の値をシャッフルして予測
        X_permuted[feature_to_permute] = X_permuted[feature_to_permute].sample(frac=1).reset_index(drop=True)
        y_pred = self.estimator.predict(X_permuted)
        

        return mean_squared_error(self.y, y_pred, squared=False)

    def pfi(self, n_shuffle: int = 10) -> None:
        """PFIを求める

        Args:
            n_shuffle: シャッフルの回数。多いほど値が安定する。デフォルトは10回
        """

        J = self.X.shape[1]  # 特徴量の数

        # J個の特徴量に対してPFIを求めたい
        # n_shuffle回シャッフルを繰り返して平均をとることで値を安定させている
        metrics_permuted = [
            np.mean(
                [self._permutation_metrics(fn) for r in range(n_shuffle)]
            )
            for fn in self.var_names
        ]

        # データフレームとしてまとめる
        # シャッフルでどのくらい予測精度が落ちるかは、差(difference)と比率(ratio)の2種類を用意する
        df_feature_importance = pd.DataFrame(
            data={
                "var_name": self.var_names,
                "baseline": self.baseline,
                "permutation": metrics_permuted,
                "difference": metrics_permuted - self.baseline,
                "ratio": metrics_permuted / self.baseline,
            }
        )

        self.feature_importance = df_feature_importance.sort_values(
            "permutation", ascending=False
        )

    def plot(self, importance_type: str = "difference") -> None:
        """PFIを可視化

        Args:
            importance_type: PFIを差(difference)と比率(ratio)のどちらで計算するか
        """

        fig, ax = plt.subplots()
        ax.barh(
            self.feature_importance["var_name"],
            self.feature_importance[importance_type],
            label=f"baseline: {self.baseline:.2f}",
        )
        ax.set(xlabel=importance_type, ylabel=None)
        ax.invert_yaxis() # 重要度が高い順に並び替える
        ax.legend(loc="lower right")
        fig.suptitle(f"Permutationによる特徴量の重要度({importance_type})")
        
        fig.show()

        
# テスト------------------------------------------------------------------------------------------------------------------------
# Random Forestの予測モデルを構築
rf = RandomForestRegressor(n_jobs=-1, random_state=42).fit(X_train, y_train)
# 予測精度を確認
print(f"R2: {r2_score(y_test, rf.predict(X_test)):.2f}")

# PFIを計算して可視化
# PFIのインスタンスの作成
pfi = PFI(rf, X_test, y_test, X_test.columns)
# PFIを計算
pfi.pfi()
# PFIを可視化
pfi.plot(importance_type="difference")
