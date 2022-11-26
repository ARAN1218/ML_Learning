class PDPICE(ICE, PDP):
    """PDPとICEを同時にプロットする"""
    
    def pdpice(self, var_name, n_grid=50):
        """PDPとICEの値を計算する
        
        Args:
            var_name:
                PDP・ICEを計算したい変数名
            n_grid: 
                グリッドを何分割するか
                細かすぎると値が荒れるが、粗すぎるとうまく関係をとらえられない
                デフォルトは50
        """
        
        # iceメソッドでiceの値を計算する。
        self.ice(var_name, range(len(self.X)), n_grid=n_grid)

        # PDPをデータフレームとしてまとめる
        self.df_pd = self.df_ice.groupby("X1").agg({"ice":"mean"}).rename(columns={"ice":"avg_pred"})

        
    def plot(self, ylim=None):
        """pdpとiceを同時にプロットする
        
        Args:
            ylim: Y軸の範囲。特に指定しなければpdpiceの範囲となる。
        """

        fig, ax = plt.subplots()
        # ICEの線
        sns.lineplot(
            self.target_var_name,
            "ice",
            units="instance",
            data=self.df_ice,
            lw=0.8,
            alpha=0.5,
            estimator=None,
            zorder=1,  # zorderを指定することで、線が背面、点が前面にくるようにする
            ax=ax,
            #label="ICE"
        )
        # インスタンスからの実際の予測値を点でプロットしておく
        sns.scatterplot(
            self.target_var_name, 
            "prediction", 
            data=self.df_instance, 
            zorder=2, 
            ax=ax,
        )
        
        # PDPの線
        ax.plot(
            self.df_pd.index,
            self.df_pd["avg_pred"],
            label="PDP"
        )
        
        # グラフの出力
        ax.set(
            xlabel=self.target_var_name,
            ylabel="(Average) Prediction",
            ylim=ylim
        )
        fig.suptitle(f"PDP & ICE ({self.target_var_name})")
        fig.legend(loc='upper right')
        fig.show()
        
        
# テスト---------------------------------------------------------------------------------------------------------------
# Random Forestで予測モデルを構築
rf = RandomForestRegressor(n_jobs=-1, random_state=42).fit(X_train, y_train)
# 予測精度を確認
display(regression_metrics(rf, X_test, y_test))

# PDP&ICEのインスタンスを作成
pdpice = PDPICE(rf, X_test)
# 各インスタンスについて、X1のPDP&ICEを計算
pdpice.pdpice("X1")
# 各インスタンスの特徴量と予測値を出力
display(pdpice.df_instance)
# 各インスタンスのPDP&ICEを可視化
pdpice.plot()
