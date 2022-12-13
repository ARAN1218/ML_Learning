import numpy as np
import random

# 複数モデル対応番
class Bagging:
    """バギング(アンサンブル学習)"""
    def __init__(self, models=[], ratio=1.0, is_regression=False):
        self.models = models
        self.ratio = ratio
        self.model_list = []
        self.is_regression = is_regression

    def fit(self, x, y):
        x, y = np.array(x), np.array(y).reshape((-1, 1))
        # 機械学習モデル用のデータの数
        n_sample = int(round(len(x) * self.ratio))
        for model_class, model_params in self.models:
            # 重複ありランダムサンプルで学習データへのインデックスを生成する
            index = random.choices(np.arange(len(x)), k=n_sample)
            # 新しい機械学習モデルを作成する
            model = model_class(**model_params)
            # 機械学習モデルを一つ学習させる
            model.fit(x[index], y[index])
            # 機械学習モデルを保存
            self.model_list.append(model)

    def predict(self, x):
        x = np.array(x)
        # 全ての機械学習モデルの結果をリストにする
        z = [model.predict(x).reshape((-1, 1)) for model in self.model_list]
        # リスト内の結果の平均をとって返す
        return np.mean(z, axis=0) if self.is_regression else np.around(np.mean(z, axis=0))

    def __str__(self):
        return '\n'.join([f'tree#{i+1}\n{model}' for i, model in enumerate(self.model_list)])
