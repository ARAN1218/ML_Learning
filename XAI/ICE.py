class ICE(PDP):
    """Indivudual Conditional Expectation (ICE)"""

    def ice(self, var_name, ids_to_compute, n_grid=50) -> None:
        """ICEを求める

        Args:
            var_name:
                ICEを計算したい変数名
            ids_to_compute:
                ICEを計算したいインスタンスのリスト
            n_grid: 
                グリッドを何分割するか
                細かすぎると値が荒れるが、粗すぎるとうまく関係をとらえられない
                デフォルトは50
        """
        
        # 可視化の際に用いるのでターゲットの変数名を保存
        self.target_var_name = var_name

        # ターゲットの変数を、取りうる値の最大値から最小値まで動かせるようにする
        value_range = np.linspace(
            self.X[self.target_var_name].min(),
            self.X[self.target_var_name].max(),
            num=n_grid
        )

        # インスタンスごとのモデルの予測値
        # PDの_counterfactual_prediction()をそのまま使っているので
        # 全データに対して予測してからids_to_computeに絞り込んでいるが
        # 本当は絞り込んでから予測をしたほうが速い
        individual_prediction = np.array([
            self._counterfactual_prediction(var_name, x)[ids_to_compute]
            for x in value_range
        ])

        # ICEをデータフレームとしてまとめる
        self.df_ice = (
            # ICEの値
            pd.DataFrame(data=individual_prediction, columns=ids_to_compute)
            # ICEで用いた特徴量の値。特徴量名を列名としている
            .assign(**{var_name: value_range})
            # 縦持ちに変換して完成
            .melt(id_vars=var_name, var_name="instance", value_name="ice")
        )

        # ICEを計算したインスタンスについての情報も保存しておく
        # 可視化の際に実際の特徴量の値とその予測値をプロットするために用いる
        self.df_instance = (
            # インスタンスの特徴量の値
            pd.DataFrame(
                data=self.X.iloc[ids_to_compute],
                columns=self.var_names
            )
            # インスタンスに対する予測値
            .assign(
                instance=ids_to_compute,
                prediction=self.estimator.predict(self.X.iloc[ids_to_compute]),
            )
            # 並べ替え
            .loc[:, ["instance", "prediction"] + self.var_names]
        )

    def plot(self, ylim=None) -> None:
        """ICEを可視化

        Args:
            ylim: Y軸の範囲。特に指定しなければiceの範囲となる。
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
        )
        # インスタンスからの実際の予測値を点でプロットしておく
        sns.scatterplot(
            self.target_var_name, 
            "prediction", 
            data=self.df_instance, 
            zorder=2, 
            ax=ax
        )
        ax.set(xlabel=self.target_var_name, ylabel="Prediction", ylim=ylim)
        fig.suptitle(
            f"Individual Conditional Expectation({self.target_var_name})"
        )
        
        fig.show()

        
# テスト---------------------------------------------------------------------------------------------------------------------
# Random Forestで予測モデルを構築
rf = RandomForestRegressor(n_jobs=-1, random_state=42).fit(X_train, y_train)
# 予測精度を確認
regression_metrics(rf, X_test, y_test)

# ICEのインスタンスを作成
ice = ICE(rf, X_test)
# インスタンス0~20について、X1のICEを計算
ice.ice("X1", range(20))
## インスタンス0について、X1のICEを計算
## ice.ice("X1", [0])
# インスタンス0~20の特徴量と予測値を出力
display(ice.df_instance)
# インスタンス0~20のICEを可視化
ice.plot(ylim=(-6, 6))
