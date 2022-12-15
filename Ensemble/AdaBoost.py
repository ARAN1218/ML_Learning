import numpy as np

# sklearnのdecision treeを用いる想定の汎用クラス
class AdaBoost:
    def __init__(self, boost=5, model_param={}):
        from sklearn.tree import DecisionTreeClassifier
        self.boost = boost
        self.model_param = model_param
        self.trees = None
        self.alpha = None

    def fit(self, x, y):
        if len(y.unique()) != 2:
            print("Only two classify")
            return # オリジナルのAdaBoostは2クラス分類のみ
        x, y = np.array(x), np.array(y).reshape((-1, 1))
        # ブースティングで使用する変数
        self.trees = []  # 各機械学習モデルの配列
        self.alpha = np.zeros((self.boost))  # 貢献度の配列
        y_bin = y * 2 - 1    # 1と-1の配列にする
        # 学習データに対する重み
        weights = np.ones((len(x),)) / len(x)
        # ブースティング
        for i in range(self.boost):
            # 決定木モデルを作成
            tree = DecisionTreeClassifier(**self.model_param)
            tree.fit(x, y, sample_weight=weights)
            # 一度、学習データに対して実行する
            z = tree.predict(x).reshape((-1, 1))
            z_bin = z * 2 - 1  # 1と-1の配列にする
            # 正解したデータを探す
            _filter = (z_bin == y_bin)  # 正解データの位置がTrueになる配列
            weights = weights.reshape((-1, 1))
            err = weights[_filter==False].sum()  # 不正解の位置にある重みの合計
            print(f'iter #{i+1} -- error={err}')
            # 終了条件
            if i == 0 and err == 0:  # 最初に完全に学習してしまった
                self.trees.append(tree)  # 最初のモデルだけ
                self.alpha = [1]
                break
            if err >= 0.5 or err == 0:  # 正解率が1/2を下回った
                self.alpha = self.alpha[:i]  # 一つ前まで
                break
            # 学習したモデルを追加
            self.trees.append(tree)
            # AdaBoostの計算
            self.alpha[i] = np.log((1.0 - err) / err) / 2.0 # 式9
            weights *= np.exp(-1.0 * self.alpha[i] * y_bin * z_bin) # 式10
            weights /= weights.sum() # 重みの正規化
            weights = weights.squeeze()

    def predict(self, x):
        x = np.array(x)
        # 各モデルの出力の合計
        z = np.zeros((len(x))).reshape((-1, 1))
        for i, tree in enumerate(self.trees):
            p = tree.predict(x).reshape((-1, 1))
            p_bin = p * 2 - 1    # 1と-1の配列にする
            z += p_bin * self.alpha[i]    # 貢献度を加味して追加
        # 合計した出力を、その符号で[0,1]と[1,0]の配列にする
        return np.array([z > 0]).astype(int).reshape((-1, 1))

    def __str__(self):
        s = []
        for i, t in enumerate(self.trees):
            s.append(f'tree: #{i+1} -- weight={self.alpha[i]}')
            s.append(str(t))
        return '\n'.join(s)
