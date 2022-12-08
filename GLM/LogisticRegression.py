class LogisticRegression:
    """ロジスティック回帰"""
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
        print(f"ロジスティック回帰による学習を開始します(epoch={self.epoch}、eta={self.eta})")
        
        self.X = self.create_matrix(self.standardize(X))
        self.Y = np.array(Y.values.T)[0]
        self.parameter = np.random.rand(self.X.shape[1])

        for eph in range(1, self.epoch+1): # 学習をloop回繰り返す
            # 重みベクトルの更新
            self.parameter = self.parameter - self.eta*np.dot(self.sigmoid(self.X, self.parameter)-self.Y, self.X)
            # 最初の1回と以降100回ごとにログを出力
            if (eph == 1 or eph % 100 == 0):
                y_pred = np.where(self.sigmoid(self.X, self.parameter)>=0.5, 1, 0)
                print(f'epoch:{eph}  accuracy:{self.accuracy(self.Y, y_pred)}')
        
    def predict(self, inputs, threshold=0.5):
        """予測する
        
        Args:
            inputs: 予測対象データ(pd.DataFrame)
            threshold: 閾値(default:0.5)(float)
        """
        inputs = self.create_matrix(self.standardize(inputs))
        result = np.where(self.sigmoid(inputs, self.parameter)>=threshold, 1, 0)
        return result
