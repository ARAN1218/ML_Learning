class MRA:
    """重回帰分析"""
    
    def create_matrix(self, X):
        """教師データにバイアスに対応するx0を加えた行列を作成
        
        Args:
            X: 教師データ(pd.DataFrame)
        """
        x0 = np.ones([X.shape[0], 1]) # バイアスに対応する1の項
        return np.hstack([x0, X])
    
    def fit(self, X, Y):
        """学習する
        
        Args:
            X: 教師データ(pd.DataFrame)
            Y: 正解データ(pd.DataFrame)
        """
        X = self.create_matrix(X)
        self.parameter = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
        
    def predict(self, X):
        """予測する
        
        Args:
            X: 予測データ(pd.DataFrame)
        """
        X = self.create_matrix(X)
        pred = np.dot(X, self.parameter)
        return pred
