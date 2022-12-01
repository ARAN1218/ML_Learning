class PERCEPTRON:
    """単純パーセプトロンクラス"""
    
    def __init__(self, epoch=100, eta=0.001):
        """単純パーセプトロンに用いるパラメータを設定する
        
        Args:
            epoch: 繰り返し回数(int)
            eta: 学習率(float)
        """
        self.epoch = epoch
        self.eta = eta
    
    def inputsum(self, inputs, weights):
        """入力データと重みベクトルの内積を求める

        Args:
            inputs: 入力データ(list[flaot])
            weights: 重みベクトル(list[float])
        """
        return np.dot(inputs, weights)

    def output(self, x):
        """単純パーセプトロンの活性化関数(ステップ関数)

        Args:
            x: 単純パーセプトロンの計算結果(float)
        """
        def step(x):
            return 1 if x > 0 else 0

        return step(x)

    def error(self, output, label):
        """予測と正解との差を求める

        Args:
            output: 予測データ(int)
            label: 正解データ(int)
        """
        return label - output

    def update(self, weights, err, x, eta=0.001):
        """予測誤差の向きに応じて重みベクトルを更新する

        Args:
            weights: 重みベクトル(list[float])
            err: 予測誤差(int)
            x: 予測に失敗した学習データ(list[float])
            eta: 学習率(float)
        """
        for i in range(len(x)):
            weights[i] += eta * float(err) * x[i]
    
    def fit(self, X, Y):
        """単純パーセプトロンアルゴリズムでデータを学習する

        Args:
            X: 教師データ(pd.DataFrame)
            Y: 正解データ(pd.DataFrame)
        """
        
        print(f"単純パーセプトロンによる学習を開始します(epoch={self.epoch}、eta={self.eta})")
        self.X = X
        self.Y = Y
        self.weights = np.random.rand(len(self.X.columns))
        flag = False
        
        for eph in range(1, self.epoch+1):
            flag = True
            sumE = 0
            
            for i, x in enumerate(self.X.itertuples()):
                x = x[1:]
                e = self.error(self.output(self.inputsum(x, self.weights)), self.Y.iloc[i][0])
                sumE += e**2
                if e != 0:
                    self.update(self.weights, e, x, eta=self.eta)
                    flag = False
            
            print(f"epoch: {eph}/{self.epoch}  accuracy: {1 - sumE/len(self.X)}")
            if flag == True: break
        
    def predict(self, inputs):
        """線形パーセプトロンで入力したデータを分類する
        
        Args:
            inputs: 予測対象データ(pd.DataFrame)
        """
        
        self.df_result = inputs.copy()
        self.df_result['pred'] = [self.output(self.inputsum(x[1:], self.weights)) for x in self.df_result.itertuples()]
        
        return self.df_result['pred']
