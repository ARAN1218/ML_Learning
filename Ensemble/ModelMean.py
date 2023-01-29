import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

class Gating:
    """ゲーティング"""
    
    def __init__(self, is_regression=False, models=[], K=4):
        """初期化
        
        Args:
            is_regression: 分類か回帰か(boolean)
                True: 回帰
                False: 分類
            models: 検討する機械学習モデルのインスタンスが入ったリスト(list[ABCMeta])
            K: 交差検証のデータ分割数(int)
        """
        self.is_regression = is_regression
        self.selected = None
        self.clf = models
        self.K = K
        self.perceptron = None
        
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
        """ゲーティングを用いてモデルを集約する
        
        Args:
            x: 学習データ(pd.DataFrame)
            y: 正解データ(pd.DataFrame)
        """
        x = np.array(x)
        y = np.array(y).reshape((-1, 1))
        
        # 交差検証を行う
        predicts = self.cv(x, y)
        n_fold = len(predicts) // len(self.clf)
        sp_data = np.zeros((x.shape[0], y.shape[1], len(self.clf)))
        for i in range(n_fold):
            for j in range(len(self.clf)):
                # テスト用データに対する結果を整形しておく
                p = predicts.pop(0)
                sp_data[p[2],:,j] = p[0]
        
        # パーセプトロンを学習させる
        self.perceptron = []
        for k in range(y.shape[1]):
            px = sp_data[:,k,:]
            py = y[:,k]
            ln = LinearRegression() if self.is_regression else LogisticRegression()
            ln.fit(px, py)
            self.perceptron.append(ln)
        
        # 全てのモデルに全てのデータを学習させる
        for j in range(len(self.clf)):
            self.clf[j].fit(x, y)
        return self

    def predict(self, x):
        """モデルを実行し、結果を集約する
        
        Args:
            x: 予測データ(pd.DataFrame)
        """
        x = np.array(x)
        # 全てのモデルを実行する
        sp_data = np.zeros((x.shape[0], len(self.perceptron), len(self.clf)))
        for j in range(len(self.clf)):
            sp_data[:,:,j] = self.clf[j].predict(x).reshape((-1,1))
        # それぞれのモデルの出力を最終モデルで合算する
        result = np.zeros((x.shape[0], len(self.perceptron)))
        for k in range(len(self.perceptron)):
            px = sp_data[:,k,:]
            result[:,k] = self.perceptron[k].predict(px).reshape((-1,))
        # 結果を返す
        return result

    def __str__(self):
        """モデルの開示"""
        return '\n'.join([str(p) for p in self.perceptron])

      
      
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

class Stacking:
    """スタッキング
    
    level1までだとブレンディングとも言う
    """
    
    def __init__(self, is_regression=False, models=[]):
        """初期化
        
        Args:
            is_regression: 分類か回帰か(boolean)
                True: 回帰
                False: 分類
            models: 検討する機械学習モデルのインスタンスが入ったリスト(list[ABCMeta])
        """
        self.is_regression = is_regression
        self.clfs = models

    def fit(self, x, y):
        """スタッキング(ブレンディング)を用いてモデルを集約する
        
        Args:
            x: 学習データ(pd.DataFrame)
            y: 正解データ(pd.DataFrame)
        """
        x = np.array(x)
        y = np.array(y).reshape((-1, 1))
        
        # level毎に学習・予測を行い、それらを次のlevelのモデルの説明変数として繰り返し学習する
        current_x = x.copy()
        for clfs in self.clfs:
            next_x = np.zeros([x.shape[0], len(clfs)])
            for i,clf in enumerate(clfs):
                clf.fit(current_x,y)
                next_x[:,i] = clf.predict(current_x).T
            current_x = next_x.copy()
        
        return self

    def predict(self, x):
        """モデルを実行し、結果を集約する
        
        Args:
            x: 予測データ(pd.DataFrame)
        """
        x = np.array(x)
        
        # fitと同様の流れでモデルのスタッキングを行う
        current_x = x.copy()
        for clfs in self.clfs:
            next_x = np.zeros([x.shape[0], len(clfs)])
            for i,clf in enumerate(clfs):
                clf.fit(current_x,y)
                next_x[:,i] = clf.predict(current_x).T
            current_x = next_x.copy()
        
        # 結果を返す
        return next_x

    def __str__(self):
        """モデルの開示"""
        stack_models = []
        for i,clfs in enumerate(self.clfs):
            stack_models.append(f"level{i}:")
            for clf in clfs:
                stack_models.append(str(clf))
        
        return '\n'.join(stack_models)



import numpy as np
import pandas as pd
from sklearn import tree

class NFoldMean:
    """NFold平均"""
    
    def __init__(self, is_regression=False, model=None, K=4):
        """初期化
        
        Args:
            is_regression: 分類か回帰か(boolean)
                True: 回帰
                False: 分類
            models: 検討する機械学習モデルのインスタンスが入ったリスト(list[ABCMeta])
            K: 交差検証のデータ分割数(int)
        """
        self.is_regression = is_regression
        self.K = K
        self.clf = model if model is not None else \
                    tree.DecisionTreeClassifier() if not is_regression else \
                    tree.DecisionTreeRegressor()
        self.clf = [self.clf for _ in range(self.K)]

    def fit(self, x, y):
        """NFold平均を用いてモデルを集約する
        
        Args:
            x: 学習データ(pd.DataFrame)
            y: 正解データ(pd.DataFrame)
        """
        x = np.array(x)
        y = np.array(y).reshape((-1, 1))
        
        # 交差検証による選択
        perm_indexs = np.random.permutation(x.shape[0])
        indexs = np.array_split(perm_indexs, self.K)
        # 交差検証を行う
        for i in range(self.K):
            # 学習用データを分割する
            ti = list(range(self.K))
            ti.remove(i)
            train = np.hstack([indexs[t] for t in ti])
            test = indexs[i]
            # 分割したデータで学習
            self.clf[i].fit(x[train], y[train])
        return self

    def predict(self, x):
        """モデルを実行し、結果を集約する
        
        Args:
            x: 予測データ(pd.DataFrame)
        """
        # 全てのモデルを実行する
        result = []
        for j in range(len(self.clf)):
            result.append(self.clf[j].predict(x))
        # 全てのモデルの平均を返す
        if self.is_regression:
            return np.array(result).mean(axis=0)
        else:
            result = np.array(result).mean(axis=0)
            return np.round(result, decimals=0)

    def __str__(self):
        """モデルの開示"""
        return '\n'.join([str(c) for c in self.clf])



import numpy as np
import pandas as pd

class SmoothedICMean:
    """スムース情報量基準平均"""
    
    def __init__(self, isregression=False, models=[], K=5, ic="AIC"):
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
        self.ic_scores = None
        
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
            train = list(np.hstack([indexs[t] for t in ti]))
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
        """スムース情報量基準平均を用いてモデルを集約する
        
        Args:
            x: 学習データ(pd.DataFrame)
            y: 正解データ(pd.DataFrame)
        """
        x = np.array(x)
        y = np.array(y).reshape((-1, 1))
        
        # 交差検証を行う
        self.ic_scores = np.zeros((len(self.clf),))
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
                    self.ic_scores[j] += 2 * np.log(score + 1e-9) + 2 * n_params
        else: # 情報量基準としてBICを選択
            for i in range(n_fold):
                for j in range(len(self.clf)):
                    # 評価スコアを尤度関数の代わりに使用する
                    p = predicts.pop(0)
                    score = self.metric(p[0], p[1])
                    # モデルのパラメータ数を取得する
                    n_params = x.shape[1]
                    # 罰則項を加えたスコアで計算
                    self.ic_scores[j] += x.shape[0] * np.log(score + 1e-9) + n_params * np.log(x.shape[0])
        
        # 情報量基準のスコアを保存しておく
        self.scores = [str(clf)+f"　{self.ic}："+str(score) for clf,score in zip(self.clf,self.ic_scores)]

        # 全てのモデルを学習させる
        for j in range(len(self.clf)):
            self.clf[j].fit(x, y)
        return self

    def predict(self, x):
        """モデルを実行し、結果を集約する
        
        Args:
            x: 予測データ(pd.DataFrame)
        """
        # スコアは小さい方が良い値なので、最大値から引く
        scores = np.max(self.ic_scores) - self.ic_scores
        # 合算が1になるようにする
        weights = scores / np.sum(scores)
        # 全てのモデルを実行する
        result = []
        for j in range(len(self.clf)):
            result.append(self.clf[j].predict(x).reshape((-1,1)))
        # 全てのモデルの重み付き平均を返す
        if self.is_regression:
            return np.average(np.array(result), axis=0, weights=weights)
        else:
            result = np.average(np.array(result), axis=0, weights=weights)
            return np.round(result, decimals=0)
    
    def __str__(self):
        """モデルの開示"""
        return "\n".join(self.scores)
