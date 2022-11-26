class PDP:
    """Partial Dependence Plot (PDP)

    Args:
        estimator: 学習済みモデル(Any)
        X: 特徴量(pd.DataFrame)
    """
    
    def __init__(self, estimator, X):
        self.estimator = estimator
        self.X = X
        self.var_names = list(self.X.columns)
    
    def _counterfactual_prediction(self, var_name_to_replace: str, value_to_replace: float) -> pd.DataFrame:
        """ある特徴量の値を置き換えたときの予測値を求める

        Args:
            var_name_to_replace: 値を置き換える特徴量の名前
            value_to_replace: 置き換える値
        """

        # 特徴量の値を置き換える際、元データが上書きされないようコピー
        X_replaced = self.X.copy()

        # 特徴量の値を置き換えて予測
        X_replaced[var_name_to_replace] = value_to_replace
        y_pred = self.estimator.predict(X_replaced)

        return y_pred

    def pdp(self, var_name: str, n_grid: int = 50) -> None:
        """PDを求める

        Args:
            var_name: 
                PDを計算したい特徴量の名前
            n_grid: 
                グリッドを何分割するか
                細かすぎると値が荒れるが、粗すぎるとうまく関係を捉えられない
                デフォルトは50
        """
        
        # 可視化の際に用いるのでターゲットの変数名を保存
        self.target_var_name = var_name

        # ターゲットの変数を、取りうる値の最大値から最小値まで動かせるようにする
        value_range = np.linspace(
            self.X[var_name].min(), 
            self.X[var_name].max(), 
            num=n_grid
        )

        # インスタンスごとのモデルの予測値を平均
        average_prediction = np.array([
            self._counterfactual_prediction(var_name, x).mean()
            for x in value_range
        ])

        # データフレームとしてまとめる
        self.df_partial_dependence = pd.DataFrame(
            data={var_name: value_range, "avg_pred": average_prediction}
        )

    def plot(self, ylim=None) -> None:
        """PDを可視化

        Args:
            ylim: 
                Y軸の範囲
                特に指定しなければavg_predictionの範囲となる
                異なる特徴量のPDを比較したいときなどに指定する
        """

        fig, ax = plt.subplots()
        ax.plot(
            self.df_partial_dependence[self.target_var_name],
            self.df_partial_dependence["avg_pred"],
        )
        ax.set(
            xlabel=self.target_var_name,
            ylabel="Average Prediction",
            ylim=ylim
        )
        fig.suptitle(f"Partial Dependence Plot ({self.target_var_name})")
        
        fig.show()

        
# テスト--------------------------------------------------------------------------------------------------------------------
# シミュレーションデータの生成
X_train, X_test, y_train, y_test = generate_simulation_data2()
# Random Forestによる予測モデルの構築
rf = RandomForestRegressor(n_jobs=-1, random_state=42).fit(X_train, y_train)

# PDのインスタンスを作成
pdp = PDP(rf, X_test)
# X1に対するPDを計算
pdp.pdp("X1", n_grid=50)
# PDを可視化
pdp.plot()
