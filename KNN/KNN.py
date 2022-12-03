import numpy as np
import pandas as pd

class KNN:
    """K近傍法クラス"""
    
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
        self.cov_inv = np.linalg.pinv(self.X.cov())
        self.mean = pd.DataFrame(self.X.mean()).T
        self.target_name = self.Y.columns[0]
    
    def euclid(self, inputs):
        """ユークリッド距離"""
        return (((self.X - inputs)**2).sum(axis=1))**0.5
    
    def manhattan(self, inputs):
        """マンハッタン距離"""
        return abs(self.X - inputs)
    
    def mahalanobis(self, inputs):
        """マハラノビス距離"""
        # 注意：平均の代わりに学習データのインスタンスを採用しているため、厳密なマハラノビス距離の定義とは異なる。
        return self.X.apply(lambda x : (np.dot((inputs - x), np.dot(self.cov_inv, (inputs - x))))**0.5, axis=1)
    
    def chebyshev(self, inputs):
        """チェビシェフ距離"""
        return self.X.apply(lambda x : max(x - inputs), axis=1)
        
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
        d_list = []
        
        for row in input_data.itertuples():
            index = row[0]
            row = row[1:]
            
            # 類似度を計算
            self._X[method] = self.euclid(row) if method=="Euclid" \
                            else self.manhattan(row) if method=="Manhattan" \
                            else self.chebyshev(row) if method=="Chebyshev" \
                            else self.mahalanobis(row)
            
            if self.types == "classification":
                # 計算した類似度を基に降順にデータを並べ替え、Kの数だけ選択して多数決投票を行う。
                pred = pd.concat([self._X, self.Y], axis=1).sort_values(method, ascending=True)[self.target_name][:self.K].value_counts()
                d = {'index':index, 'prediction':pred.index[0], 'cnt':pred[0]}
                d_list.append(d)
                
            else:
                # 計算した類似度を基に降順にデータを並べ替え、Kの数だけ選択して平均値を求める。
                pred = pd.concat([self._X, self.Y], axis=1).sort_values(method, ascending=True)[self.target_name][:self.K]
                d = {'index':index, 'prediction':pred.mean()}
                d_list.append(d)
        
        self.summary = pd.DataFrame(d_list)
        return self.summary['prediction']
        
