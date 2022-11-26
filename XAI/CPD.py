class CPD(PDP):
    """Conditional Partial Dependence (CPD)"""
        
    def _counterfactual_prediction(self, var_name_to_replace: str, value_to_replace: float, category: str) -> pd.DataFrame:
        """ある特徴量の値を置き換えたときの予測値を求める

        Args:
            var_name_to_replace: 値を置き換える特徴量の名前
            value_to_replace: 置き換える値
            category: 分割するcategoryの群の要素
        """

        # 特徴量の値を置き換える際、元データが上書きされないようコピー
        X_replaced = self.X.copy()
        # 分割カテゴリーで分ける
        X_replaced = X_replaced[X_replaced[self.category] == category]

        # 特徴量の値を置き換えて予測
        X_replaced[var_name_to_replace] = value_to_replace
        y_pred = self.estimator.predict(X_replaced)

        return y_pred
        
    def cpd(self, var_name, category, n_grid=50):
        """cpdを計算する"""
        self.target_var_name = var_name
        self.category = category
        self.df_cpds = []
            
        # ターゲットの変数を、取りうる値の最大値から最小値まで動かせるようにする
        value_range = np.linspace(
            self.X[self.target_var_name].min(),
            self.X[self.target_var_name].max(),
            num=n_grid
        )

        # インスタンスごとのモデルの予測値
        for c in self.X[self.category].unique():
            conditional_prediction = np.array([
                self._counterfactual_prediction(var_name, x, c).mean()
                for x in value_range
            ])
            
            # データフレームとしてまとめる
            df_cpd = pd.DataFrame(
                data={var_name: value_range, "avg_pred": conditional_prediction}
            )
            self.df_cpds.append(df_cpd.copy())
        
    def plot(self, ylim=None):
        """cpdを可視化する
        
        Args:
            ylim: Y軸の範囲。特に指定しなければcpdの範囲となる。
        """

        fig, ax = plt.subplots()
        
        # CDPの線
        for i, c in enumerate(self.X[self.category].unique()):
            ax.plot(
                self.df_cpds[i][self.target_var_name],
                self.df_cpds[i]["avg_pred"],
                label=c
            )
        
        # グラフの出力
        ax.set(
            xlabel=self.target_var_name,
            ylabel="Average Prediction",
            ylim=ylim
        )
        fig.suptitle(f"Conditional Partial Dependence ({self.target_var_name})")
        fig.legend(loc='upper right', title=self.category)
        fig.show()

        
# テスト----------------------------------------------------------------------------------------------------------------
# Random Forestで予測モデルを構築
rf = RandomForestRegressor(n_jobs=-1, random_state=42).fit(X_train, y_train)
# 予測精度を確認
display(regression_metrics(rf, X_test, y_test))

# CPDのインスタンスを作成
cpd = CPD(rf, X_test)
# X2をカテゴリーとしてX1のCPDを計算
cpd.cpd("X1", "X2")
# CPDを可視化
cpd.plot()
