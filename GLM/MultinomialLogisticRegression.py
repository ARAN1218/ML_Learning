class MLR:
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
    
    def create_onehot_matrix(self, Y):
        """正解データを要素毎にOheHotEncodingする
        
        Args:
            Y: 正解データ(pd.DataFrame)
        """
        self.Y_pre = np.array(Y[Y.columns[0]])
        Y_get_dummies = pd.get_dummies(Y[Y.columns[0]])
        self.target_columns = Y_get_dummies.columns
        return np.array(Y_get_dummies)
    
    def accuracy(self, y_true, y_pred):
        """正答率算出関数
        
        Args:
            y_true: 正解データ(np.array)
            y_pred: 予測値(np.array)
        """
        return np.sum(y_true==y_pred)/len(y_true)
    
    def softmax(self, X):
        """ソフトマックス関数
        
        Args:
            X: 学習データ(np.array)
            parameter: 重みベクトル(np.array)
        """
        exp_x = np.exp(X-np.max(X, axis=1).reshape(X.shape[0],-1)) # オーバーフローを防止する
        y = exp_x / np.sum(exp_x, axis=1)[:, np.newaxis]
        return y
        
    def fit(self, X, Y, standardize=False):
        """学習する
        
        Args:
            X: 教師データ(pd.DataFrame)
            Y: 正解データ(pd.DataFrame)
        """
        print(f"多項ロジスティック回帰による学習を開始します(epoch={self.epoch}、eta={self.eta})")
        
        self.Y = self.create_onehot_matrix(Y)
        self.X = self.create_matrix(self.standardize(X))
        self.parameter = np.random.randn(self.X.shape[1], self.Y.shape[1])
        
        for eph in range(1, self.epoch+1): # 学習をepoch回繰り返す
            # 重みベクトルの更新
            self.parameter = self.parameter - self.eta*np.dot(self.X.T, self.softmax(np.dot(self.X, self.parameter))-self.Y) / (self.X.shape[0]-1)
            # 最初の1回と以降100回ごとにログを出力
            if (eph == 1 or eph % 100 == 0):
                print(f'eposh: {eph}  accurary:{self.accuracy(self.Y_pre, self.softmax(np.dot(self.X, self.parameter)).argmax(axis=1))}')
    
    def predict(self, inputs):
        """予測する
        
        Args:
            inputs: 予測対象データ(pd.DataFrame)
        """
        inputs = self.create_matrix(self.standardize(inputs))
        result = pd.DataFrame(self.softmax(np.dot(inputs, self.parameter)), columns=self.target_columns)
        result['prediction'] = result.apply(lambda x : x[x==max(x)].index[0], axis=1)
        return result
