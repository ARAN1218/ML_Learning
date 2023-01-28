import numpy as np
import pandas as pd
import random
from sklearn import tree


class Bumping:
    """バンピングによるモデル選択"""
    
    def __init__(self, is_regression=False, tree=None, tree_params={}, K=4, n_trees=5, ratio=1.0):
        """初期化
        
        Args:
            is_regression: 分類か回帰か(boolean)
                True: 回帰
                False: 分類
            tree: 使用する機械学習モデルのクラス(ABCMeta)
            tree_params: treeで設定した機械学習モデルのパラメータ(dict)
            K: 交差検証のデータ分割数(int)
            n_trees: 作成する機械学習モデルの数(int)
            ratio: ブートストラップによるデータ生成の割合(float)
        """
        self.is_regression = is_regression
        self.selected = None
        self.tree = tree if tree is not None else \
                    tree.DecisionTreeClassifier if not is_regression else \
                    tree.DecisionTreeRegressor
        self.trees = None
        self.tree_params = tree_params
        self.K = K
        self.n_trees = n_trees
        self.ratio = ratio
        
    def metric(self, y_pred, y_true):
        """正解データとの差をスコアにする関数"""
        s = np.array([])
        if self.is_regression:  # 回帰
            s = (y_pred - y_true)**2  # 二乗誤差
        else:  # クラス分類
            # 値が小さいほど良いので不一致の数（1−accuracy）
            s = (y_pred != y_true).astype(np.float32)
        return s.mean()  # 平均値を返す

    def cv(self, x, y):
        """交差検証による選択"""
        predicts = []
        
        # シャッフルしたインデックスを交差検証の数に分割する
        perm_indexs = np.random.permutation(x.shape[0])
        indexs = np.array_split(perm_indexs, self.K)
        
        # 交差検証を行う
        for i in range(self.K):
            # 学習用データを分割する
            ti = list(range(self.K))
            ti.remove(i)
            train = np.hstack([indexs[t] for t in ti])
            test = indexs[i]
        
            # 全てのモデルを検証する
            for j in range(self.n_trees):
                # 一度分割したデータで学習
                self.trees[j].fit(x[train], y[train])
                # 一度実行してスコアを作成
                z = self.trees[j].predict(x[test]).reshape((-1, 1))
                predicts.append((z, y[test], test))
        
        return predicts

    def fit(self, x, y):
        """バンピングを用いてモデルを比較・選択する
        
        Args:
            x: 学習データ(pd.DataFrame)
            y: 正解データ(pd.DataFrame)
        """
        x, y = np.array(x), np.array(y).reshape((-1,1))
        self.trees = []
        # 機械学習モデル用のデータの数
        n_sample = int(round(len(x) * self.ratio))
        for _ in range(self.n_trees):
            # 重複ありランダムサンプルで学習データへのインデックスを生成する
            index = random.choices(np.arange(len(x)), k=n_sample)
            # 新しい機械学習モデルを作成する
            tree = self.tree(**self.tree_params)
            # 機械学習モデルを一つ学習させる
            tree.fit(x[index], y[index])
            # 機械学習モデルを保存
            self.trees.append(tree)
            
        # 作ったモデルを評価する
        scores = np.zeros((self.n_trees,))
        predicts = self.cv(x, y)
        K = len(predicts) // self.n_trees
        
        # 交差検証の結果を取得
        for i in range(K):
            for j in range(len(self.trees)):
                p = predicts.pop(0)
                scores[j] += self.metric(p[0], p[1])
        
        # 最終的に最も良いモデルを選択
        self.selected = self.trees[np.argmin(scores)]
        return self
    
    def predict(self, x):
        """選択されたモデルを実行
        
        Args:
            x: 予測データ(pd.DataFrame)
        """
        x = np.array(x)
        return self.selected.predict(x)

    def __str__(self):
        """選択したモデルの開示"""
        return str(self.selected)
    
