import numpy as np
import random

class RandomForest:
    # self, features=5, max_depth=5, metric=entropy.gini, leaf=ZeroRule, tree_params={} depth=1, is_regression=False
    def __init__(self, max_features=5, n_trees=5, ratio=1.0, tree_params={}, is_regression=False):
        from sklearn import tree
        self.tree = tree.DecisionTreeClassifier if not is_regression else \
                    tree.DecisionTreeRegressor
        self.trees = []
        self.indexes = []
        self.tree_params = tree_params
        self.max_features = max_features
        self.n_trees = n_trees
        self.ratio = ratio
        self.is_regression = is_regression

    def fit(self, x, y):
        """基本的にバギングだが、使用する特徴量を選定する工程が入っている"""
        x, y = np.array(x), np.array(y)#.reshape((-1, 1))
        # 説明変数内の次元から、ランダムに使用する次元を選択する
        for _ in range(self.n_trees):
            n_sample = int(round(len(x) * self.ratio))
            index_features = random.sample(range(x.shape[1]), random.randint(1, min(self.max_features, x.shape[1])))
            # 重複ありランダムサンプルで学習データへのインデックスを生成する
            index_data = random.choices(np.arange(len(x)), k=n_sample)
            # 新しい機械学習モデルを作成する
            tree = self.tree(**self.tree_params)
            # 機械学習モデルを一つ学習させる
            #print(x[index_data][:,index_features])
            #print(y[index_data])
            tree.fit(x[index_data][:,index_features], y[index_data])
            # 機械学習モデルを保存
            self.trees.append(tree)
            self.indexes.append(index_features)
    
    def predict(self, x):
        x = np.array(x)
        # 全ての機械学習モデルの結果をリストにする
        z = [tree.predict(x[:,index]) for tree,index in zip(self.trees, self.indexes)]
        # リスト内の結果の平均をとって返す
        return np.mean(z, axis=0) if self.is_regression else np.around(np.mean(z, axis=0))
    
    def __str__(self):
        return '\n'.join([f'tree#{i+1}\n{model[0]}　index_features: {model[1]}' for i, model in enumerate(zip(self.trees, self.indexes))])
