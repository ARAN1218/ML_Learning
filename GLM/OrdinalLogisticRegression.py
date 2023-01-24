class OLR:
    """順序ロジスティック回帰"""
    def __init__(self, eta=0.001, epoch=1000):
        """データを初期化する
        
        Args:
            eta: 学習率(default:0.001)(float)
            epoch: 更新回数(default:1000)(int)
        """
        self.eta = eta
        self.epoch = epoch
        
    def standardize(self, x):
        """標準化を行う
        
        Args:
            x: 標準化前の学習データ(pd.DataFrame)
        """
        x_mean = x.mean(axis=0)       # 列ごとの平均値を求める
        std = x.std(axis=0)           # 列ごとの標準偏差を求める  
        return (x - x_mean) / std     # 標準化した値を返す
    
    def create_matrix(self, X):
        """学習データにバイアスに対応するx0を加えた行列を作成
        
        Args:
            X: 学習データ(pd.DataFrame)
        """
        x0 = np.ones([X.shape[0], 1]) # バイアスに対応する1の項
        return np.hstack([x0, X])
    
    def accuracy(self, y_true, y_pred):
        """正答率算出関数
        
        Args:
            y_true: 正解データ(np.array)
            y_pred: 予測値(np.array)
        """
        return np.sum(y_true==y_pred)/len(y_true)
        
    def sigmoid(self, X, parameter):
        """シグモイド関数
        
        Args:
            X: 学習データ(np.array)
            parameter: 重みベクトル(np.array)
        """
        return 1 / (1 + np.exp(-np.dot(X, parameter)))
        
    def fit(self, X, Y):
        """学習する
        
        Args:
            X: 教師データ(pd.DataFrame)
            Y: 正解データ(pd.DataFrame)
        """
        print(f"順序ロジスティック回帰による学習を開始します(epoch={self.epoch}、eta={self.eta})")
        
        # 初期化フェーズ
        self.X = self.create_matrix(self.standardize(X))
        self.Y = Y.copy()
        self.y_list = sorted(Y[Y.columns[0]].unique())
        y_str = "".join([str(i) for i in self.y_list])
        y_str_len = len(y_str)
        self.intercepts = np.array([])
        self.parameter = np.random.rand(self.X.shape[1])
        
        for vs in range(1,len(y_str)):
            self.Y[y_str[:vs+1]+' vs '+y_str[vs:]] = Y[Y.columns[0]].map(lambda x : 1 if x in self.y_list[vs:] else 0)
            self.intercepts = np.append(self.intercepts, np.random.rand(1))
        self.Y = np.array(self.Y.T)

        # 学習フェーズ
        for eph in range(1, self.epoch+1): # 学習をepoch回繰り返す
            # 重みベクトルの更新(「1 vs. 2&3」と「1&2 vs. 3」...の更新を続けて行う)
            for vs in range(1,y_str_len):
                self.parameter = self.parameter - self.eta*np.dot(self.sigmoid(self.X, self.parameter)-self.Y[vs], self.X)
                # 切片だけ変わる処理
                self.intercepts[vs-1] = self.parameter[0]
                self.parameter[0] = self.intercepts[vs%(y_str_len-1)]
            # 最初の1回と以降100回ごとにログを出力
            if (eph == 1 or eph % 100 == 0):
                for vs in range(len(self.intercepts)):
                    y_pred = self.predict(self.X[:,1:])
                print(f'epoch:{eph}  accuracy:{self.accuracy(self.Y[0], y_pred)}')
        
    def predict(self, inputs, threshold=0.5):
        """予測する
        
        Args:
            inputs: 予測対象データ(pd.DataFrame)
            threshold: 閾値(float)(default:0.5)
        """
        inputs = self.create_matrix(self.standardize(inputs))
        result = pd.DataFrame(["" for _ in range(len(inputs))], columns=['pred'])
        result['pred'] = result['pred'].map(list)
        y_str_len = len(self.intercepts)
        
        # 「○ vs. △&✖️」のような予測結果の組み合わせをリストに代入し、最頻値を最終的な予測結果とする
        for vs in range(y_str_len):
            result['tmp'] = self.sigmoid(inputs, self.parameter)>=threshold
            result['tmp'] = result['tmp'].apply(lambda x : self.y_list[vs+1:] if x else self.y_list[:vs+1])
            result['pred'] += result['tmp']
            # 切片だけ変わる処理
            self.parameter[0] = self.intercepts[(vs+1)%y_str_len]
        result['pred'] = result['pred'].map(lambda x : pd.Series(x).mode().max())
        return np.array(result['pred'])
