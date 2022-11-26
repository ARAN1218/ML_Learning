class LOCOFI:
    """Leave One Covariate Out Feature Importance (LOCOFI)
    
    Args:
        estimator: 評価に用いる学習モデルインスタンス(Any)
        X: 特徴量(pd.DataFrame)
        y: 目的変数(pd.DataFrame)
    """
    
    # クラスインスタンスを定義
    def __init__(self, estimator, X, y):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.var_names = list(self.X.columns)
        
        # 削除なしの場合の予測精度
        # mean_squared_error()はsquared=TrueならMSE、squared=FalseならRMSE
        self.baseline = mean_squared_error(
            self.y, self.estimator.predict(self.X), squared=False
        )
        
    def _loco_metrics(self, feature_to_permute: str) -> float:
        """ある特徴量の値を削除したときの予測精度を求める

        Args:
            feature_to_permute: 削除する特徴量の名前
        """
        
        X_loco = self.X.drop(feature_to_permute, axis=1)
        self.estimator.fit(X_loco, self.y)
        y_pred = self.estimator.predict(X_loco)

        return mean_squared_error(self.y, y_pred, squared=False)
        
        
    def locofi(self) -> float:
        """LOCOFIを求める"""

        # LOCOFIを計算する
        metrics_loco = [self._loco_metrics(fn) for fn in self.var_names]

        # データフレームとしてまとめる
        # シャッフルでどのくらい予測精度が落ちるかは、差(difference)と比率(ratio)の2種類を用意する
        df_feature_importance = pd.DataFrame(
            data={
                "var_name": self.var_names,
                "baseline": self.baseline,
                "loco": metrics_loco,
                "difference": metrics_loco - self.baseline,
                "ratio": metrics_loco / self.baseline,
            }
        )

        self.feature_importance = df_feature_importance.sort_values(
            "loco", ascending=False
        )


    def plot(self, importance_type: str = "difference") -> None:
        """LOCOFIを可視化

        Args:
            importance_type: LOCOFIを差(difference)と比率(ratio)のどちらで計算するか
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
        fig.suptitle(f"LOCOによる特徴量の重要度({importance_type})")
        
        fig.show()

        
# テスト-----------------------------------------------------------------------------------------------------------------------
# Random Forestの予測モデルを構築
rf = RandomForestRegressor(n_jobs=-1, random_state=42).fit(X_train, y_train)
# 予測精度を確認
print(f"R2: {r2_score(y_test, rf.predict(X_test)):.2f}")

# LOCOFIを計算して可視化
# LOCOFIのインスタンスの作成
locofi = LOCOFI(rf, X_test, y_test)
# LOCOFIを計算
locofi.locofi()
# LOCOFIを可視化
locofi.plot(importance_type="difference")
