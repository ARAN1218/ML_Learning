import numpy as np
import pandas as pd

class KNN:
    """K近傍法"""
    
    def __init__(self, K=1, types="classification"):
        """K近傍法のパラメータを設定する
        
        Args:
            K: 予測時にサンプルを選ぶ数(int)
            types: 分類器の種類(str)
                classification: 分類
                regression: 回帰
        """
        if types not in ('classification', 'regression'):
            print("引数typesの値が異常です！(classification:分類、regression:回帰)")
        self.K = K
        self.types = types
    
    def fit(self, X, Y):
        """K近傍法の計算のためにデータを準備する

        Args:
            X: 教師データ(pd.DataFrame)
            Y: 正解データ(pd.DataFrame)
        """
        self.X = X
        self.Y = Y
        self.mean = self.X.mean()
        self.target_name = self.Y.columns[0]
    
    def euclid(self, inputs):
        """ユークリッド距離"""
        self._X[method] = (((self.X - inputs)**2).sum(axis=1))**0.5
        self.predict_to_similality()
    
    def manhattan(self, inputs):
        """マンハッタン距離"""
        self._X[method] = abs(self.X - inputs)
        self.predict_to_similality()
    
    def mahalanobis(self, inputs):
        """マハラノビス距離"""
        # 注意：平均の代わりに学習データのインスタンスを採用しているため、厳密なマハラノビス距離の定義とは異なる。
        #      また、負の数の平方根を取らないように、全体の絶対値に対して計算している。
        self._X[method] = abs(np.dot((inputs - self.X), np.dot(self.cov_inv, (inputs - self.X).T)))**0.5
        self.predict_to_similality()
    
    def chebyshev(self, inputs):
        """チェビシェフ距離"""
        self._X[method] = np.max((self.X - inputs), axis=1)
        self.predict_to_similality()
    
    def predict_to_similality(self):
        """類似度を基に予測結果を計算する"""
        if self.types == "classification":
            # 計算した類似度を基に降順にデータを並べ替え、Kの数だけ選択して多数決投票を行う。
            pred = pd.concat([self._X, self.Y], axis=1).sort_values(method, ascending=True)[self.target_name][:self.K].value_counts()
            self.pred_list.append(pred.index[0])

        else:
            # 計算した類似度を基に降順にデータを並べ替え、Kの数だけ選択して平均値を求める。
            pred = pd.concat([self._X, self.Y], axis=1).sort_values(method, ascending=True)[self.target_name][:self.K]
            self.pred_list.append(pred.mean())
        
    def predict(self, input_data, method="Euclid"):
        """K近傍法で入力したデータを分類する
        
        Args:
            input_data: 予測対象データ(pd.DataFrame)
            method: 類似度の計算方法(str)
                Manhattan: マンハッタン距離
                Euclid(Default): ユークリッド距離
                Mahalanobis: マハラノビス距離
                Chebyshev:チェビシェフ距離
        """
        if method not in ("Manhattan", "Euclid", "Mahalanobis", "Chebyshev"):
            print("引数methodの値が異常です！(Manhattan:マンハッタン距離、Euclid:ユークリッド距離、Mahalanobis:マハラノビス距離)、Chebyshev:チェビシェフ距離")
            return 
        
        # 元データに上書きしないようにコピーを作成する
        self._X = self.X.copy()
        self._X[method] = 0
        self.pred_list = []
        
        if method=="Euclid":
            input_data.apply(lambda x : self.euclid(x), axis=1)
        elif method=="Manhattan":
            input_data.apply(lambda x : self.manhattan(x), axis=1)
        elif method=="Chebyshev":
            input_data.apply(lambda x : self.chebyshev(x), axis=1)
        else:
            self.cov_inv = np.linalg.pinv(self.X.cov())
            input_data.apply(lambda x : self.mahalanobis(x), axis=1)
        
        return np.array(self.pred_list)
