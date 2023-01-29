import numpy as np
import pandas as pd

class ICSelect:
    """情報量基準(赤池・ベイズ)によるモデル選択"""
    
    def __init__(self, is_regression=False, models=[], K=4, ic="AIC"):
        """初期化
        
        Args:
            is_regression: 分類か回帰か(boolean)
                True: 回帰
                False: 分類
            models: 検討する機械学習モデルのインスタンスが入ったリスト(list[ABCMeta])
            K: 交差検証のデータ分割数(int)
            ic: 情報量基準の種類(str)
                AIC: 赤池情報量基準
                BIC: ベイズ情報量基準
        """
        self.is_regression = is_regression
        self.selected = None
        self.clf = models
        self.K = K
        self.ic = ic
        
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
            for j in range(len(self.clf)):
                # 一度分割したデータで学習
                self.clf[j].fit(x[train], y[train])
                # 一度実行してスコアを作成
                z = self.clf[j].predict(x[test]).reshape((-1, 1))
                predicts.append((z, y[test], test))
        
        return predicts

    def fit(self, x, y):
        """情報量基準を用いてモデルを比較・選択する
        
        Args:
            x: 学習データ(pd.DataFrame)
            y: 正解データ(pd.DataFrame)
        """
        x = np.array(x)
        y = np.array(y).reshape((-1, 1))
        # 交差検証を行う
        scores = np.zeros((len(self.clf),))
        predicts = self.cv(x, y)
        n_fold = len(predicts) // len(self.clf)
        
        # 交差検証の結果を取得
        if self.ic == "AIC": # 情報量基準としてAICを選択
            for i in range(n_fold):
                for j in range(len(self.clf)):
                    # 評価スコアを尤度関数の代わりに使用する
                    p = predicts.pop(0)
                    score = self.metric(p[0], p[1])
                    # モデルのパラメータ数を取得する
                    n_params = x.shape[1]
                    # 罰則項を加えたスコアで計算
                    scores[j] += 2 * np.log(score + 1e-9) + 2 * n_params
        else: # 情報量基準としてBICを選択
            for i in range(n_fold):
                for j in range(len(self.clf)):
                    # 評価スコアを尤度関数の代わりに使用する
                    p = predicts.pop(0)
                    score = self.metric(p[0], p[1])
                    # モデルのパラメータ数を取得する
                    n_params = x.shape[1]
                    # 罰則項を加えたスコアで計算
                    scores[j] += x.shape[0] * np.log(score + 1e-9) + n_params * np.log(x.shape[0])

        # 最終的に最も良いモデルを選択
        self.selected = self.clf[np.argmin(scores)]
        self.scores = [str(clf)+f"　{self.ic}："+str(score) for clf,score in zip(self.clf,scores)]
        self.scores.append("")
        self.scores.append("selected model："+str(self.clf[np.argmin(scores)])+f"　{self.ic}："+str(min(scores)))
        
        # 最も良いモデルに全てのデータを学習させる
        self.selected.fit(x, y)
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
        return "\n".join(self.scores)
