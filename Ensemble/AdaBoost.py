import numpy as np
import pandas as pd

class AdaBoost:
    """AdaBoost(総合)"""
    def __init__(self, boost=5, model='default', model_param={}, adatype='original', 
                threshold=0.01):
        """初期化
        
        Args:
            boost: 重みを更新するために繰り返す学習回数
            model: 使用する機械学習モデル(default:Decision Tree)
            model_param: modelで選択した機械学習モデルのパラメータ
            adatype: AdaBoostのタイプ
                original: オリジナルの二値分類アダブースト
                M1: 他クラス分類に拡張したアダブースト
                RT: 回帰に拡張したアダブースト
                R2: RTの改良型回帰アダブースト
            threshold: RTを選択した場合にのみ必要
        """
        self.model = model
        self.adatype = adatype
        self.adaboost = self.AdaBoost_original(model=self.model, boost=boost, model_param=model_param) if adatype=='original' else \
                        self.AdaBoost_M1(model=self.model, boost=boost, model_param=model_param) if adatype=='M1' else \
                        self.AdaBoost_RT(model=self.model, boost=boost, model_param=model_param, threshold=threshold) if adatype=='RT' else \
                        self.AdaBoost_R2(model=self.model, boost=boost, model_param=model_param)
        
        
    def fit(self, x, y):
        """AdaBoostによる学習"""
        print(f"{self.adatype}アダブーストによる学習を開始します")
        self.adaboost.fit(x, y)
        
    def predict(self, x):
        """AdaBoostによる予測"""
        print(f"{self.adatype}アダブーストによる予測を開始します")
        return self.adaboost.predict(x)
        
    def __str__(self):
        """学習結果の可視化"""
        return str(self.adaboost)
    
    class AdaBoost_original:
        """AdaBoost(original)"""
        def __init__(self, model='default', boost=5, model_param={}):
            """初期化
            
            Args: 
                model: 機械学習クラス(sklearn等)
                boost: ブースティングの回数
                model_param: modelのハイパーパラメータ
            """
            from sklearn.tree import DecisionTreeClassifier
            self.boost = boost
            self.model = DecisionTreeClassifier if model=='default' else model
            self.model_param = model_param
            self.trees = None
            self.alpha = None

        def fit(self, x, y):
            """ブースティングを用いてモデルを学習させる
            
            Args:
                x: 学習データ(pd.DataFrame)
                y: 正解データ(pd.DataFrame)
            """
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
                tree = self.model(**self.model_param)
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
                self.alpha[i] = np.log((1.0 - err) / err) / 2.0
                weights *= np.exp(-1.0 * self.alpha[i] * y_bin * z_bin)
                weights /= weights.sum() # 重みの正規化
                weights = weights.squeeze()

        def predict(self, x):
            """学習したモデルから予測する
            
            Args:
                x: 予測データ(pd.DataFrame)
            """
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
            """学習結果の可視化"""
            s = []
            for i, t in enumerate(self.trees):
                s.append(f'tree: #{i+1} -- weight={self.alpha[i]}')
                s.append(str(t))
            return '\n'.join(s)


    class AdaBoost_M1:
        """AdaBoost(M1)"""
        def __init__(self, model='default', boost=5, model_param={}):
            """初期化
            
            Args: 
                model: 機械学習クラス(sklearn等)
                boost: ブースティングの回数
                model_param: modelのハイパーパラメータ
            """
            from sklearn.tree import DecisionTreeClassifier
            self.boost = boost
            self.model = DecisionTreeClassifier if model=='default' else model
            self.model_param = model_param
            self.trees = None
            self.beta = None
            self.n_clz = 0  # クラスの個数

        def fit(self, x, y):
            """ブースティングを用いてモデルを学習させる
            
            Args:
                x: 学習データ(pd.DataFrame)
                y: 正解データ(pd.DataFrame)
            """
            # ブースティングで使用する変数
            self.trees = []  # 各機械学習モデルの配列
            self.beta = np.zeros((self.boost,))
            self.n_clz = len(y.unique())  # 扱うクラス数
            x, y = np.array(x), np.array(y).reshape((-1, 1))
            # 学習データに対する重み
            weights = np.ones((len(x),)) / len(x)
            # ブースティング
            for i in range(self.boost):
                # 決定木モデルを作成
                tree = self.model(**self.model_param)
                tree.fit(x, y, sample_weight=weights)
                # 一度、学習データに対して実行する
                z = tree.predict(x).reshape((-1, 1))
                # 正解したデータを探す
                _filter = z == y  # 正解データの位置がTrueになる配列
                weights = weights.reshape((-1, 1))
                err = weights[_filter==False].sum()  # 不正解の位置にある重みの合計
                print(f'iter #{i+1} -- error={err}')
                # 終了条件
                if i == 0 and err == 0:  # 最初に完全に学習してしまった
                    self.trees.append(tree)  # 最初のモデルだけ
                    self.beta = 1
                    break
                if err >= 0.5 or err == 0:  # 正解率が1/2を下回った
                    self.beta = self.beta[:i]  # 一つ前まで
                    break
                # 学習したモデルを追加
                self.trees.append(tree)
                # AdaBoost.M1の計算
                self.beta[i] = err / (1.0 - err)
                weights[_filter] *= self.beta[i]
                weights /= weights.sum() # 重みの正規化
                weights = weights.squeeze()

        def predict(self, x):
            """学習したモデルから予測する
            
            Args:
                x: 予測データ(pd.DataFrame)
            """
            x = np.array(x)
            # 各モデルの出力の合計
            z = np.zeros((len(x), self.n_clz))
            # 各モデルの貢献度を求める
            w = np.log(1.0 / self.beta)
            if w.sum() == 0:  # 完全に学習してしまいエラーが0の時
                w = np.ones((len(self.trees))) / len(self.trees)
            # 全てのモデルの貢献度付き合算
            for i, tree in enumerate(self.trees):
                p = tree.predict(x).reshape((-1, 1))  # p は分類されたクラスを表す一次元配列
                for j in range(len(x)):
                    z[j, p[j]] += w[i]  # 分類されたクラスの位置に貢献度を加算
            return z.argmax(axis=1)  # クラスの属する可能性を表す配列の内、最も可能性が高いクラスを予測値として返す

        def __str__(self):
            """学習結果の可視化"""
            s = []
            w = np.log(1.0 / self.beta) # モデルの貢献度
            if w.sum() == 0:
                w = np.ones((len(self.trees))) / len(self.trees)
            for i, t in enumerate(self.trees):
                s.append(f'tree: #{i+1} -- weight={w[i]}')
                s.append(str(t))
            return '\n'.join(s)
        
    class AdaBoost_RT:
        """AdaBoost(RT)"""
        def __init__(self, model='default', boost=5, model_param={}, threshold=0.01):
            """初期化
            
            Args: 
                model: 機械学習クラス(sklearn等)
                boost: ブースティングの回数
                model_param: modelのハイパーパラメータ
                threshold: modelの出力の残差の「正否」の閾値
            """
            from sklearn.tree import DecisionTreeRegressor
            self.boost = boost
            self.model = DecisionTreeRegressor if model=='default' else model
            self.model_param = model_param
            self.trees = None
            self.beta = None
            self.threshold = threshold

        def fit(self, x, y):
            """ブースティングを用いてモデルを学習させる
            
            Args:
                x: 学習データ(pd.DataFrame)
                y: 正解データ(pd.DataFrame)
            """
            x, y = np.array(x), np.array(y).reshape((-1, 1))
            # ブースティングで使用する変数
            _x, _y = x, y  # 引数を待避しておく
            self.trees = []
            self.beta = np.zeros((self.boost,))
            # 学習データに対する重み
            weights = np.ones((len(x),)) / len(x)
            # threshold値
            threshold = self.threshold
            # ブースティング
            for i in range(self.boost):
                # 決定木モデルを作成
                tree = self.model(**self.model_param)
                # 重み付きの機械学習モデルを代替するため、重みを確率にしてインデックスを取り出す
                all_idx = np.arange(x.shape[0])  # 全てのデータのインデックス
                p_weight = weights / weights.sum()  # 取り出す確率
                idx = np.random.choice(all_idx, size=x.shape[0], replace=True, p=p_weight)
                # インデックスの位置から学習用データを作成する
                x = _x[idx]
                y = _y[idx]
                # モデルを学習する
                tree.fit(x, y)
                # 一度、学習データに対して実行する
                z = tree.predict(x).reshape((-1, 1))
                # 値の大きさに影響されないよう、相対誤差とする
                l = np.absolute(z - y).reshape((-1,)) / y.mean()
                # 正解に相当するデータを探す
                _filter = l < threshold  # 正解に相当するデータの位置がTrueになる配列
                err = weights[_filter==False].sum()  # 不正解に相当する位置にある重みの合計
                print(f'iter #{i+1} -- error={err}')
                # 終了条件
                if i == 0 and err == 0:  # 最初に完全に学習してしまった
                    self.trees.append(tree)  # 最初のモデルだけ
                    self.beta = np.array([1])
                    break
                if err < 1e-10:  # 完全に学習してしまった
                    self.beta = self.beta[:i]
                    break
                # AdaBoost.RTの計算
                self.trees.append(tree)
                self.beta[i] = err / (1.0 - err)
                weights[_filter] *= self.beta[i] ** 2
                weights /= weights.sum() # 重みの正規化

        def predict(self, x):
            """学習したモデルから予測する
            
            Args:
                x: 予測データ(pd.DataFrame)
            """
            x = np.array(x)
            # 各モデルの出力の合計
            z = np.zeros((len(x),1))
            # 各モデルの貢献度を求める
            w = np.log(1.0 / self.beta) if len(self.beta)>1 else np.ones((len(self.trees),))
            # 全てのモデルの貢献度付き合算
            for i, tree in enumerate(self.trees):
                p = tree.predict(x).reshape((-1, 1))
                z += p * w[i]
            return z / w.sum()

        def __str__(self):
            """学習結果の可視化"""
            s = []
            w = np.log(1.0 / self.beta) if len(self.beta)>1 else np.ones((len(self.trees),))
            if w.sum() == 0:
                w = np.ones((len(self.trees),)) / len(self.trees)
            else:
                w /= w.sum()
            for i, t in enumerate(self.trees):
                s.append(f'tree: #{i+1} -- weight={w[i]}')
                s.append(str(t))
            return '\n'.join(s)
        
        
    class AdaBoost_R2:
        """AdaBoost(R2)"""
        def __init__(self, model='default', boost=5, model_param={}):
            """初期化
            
            Args: 
                model: 機械学習クラス(sklearn等)
                boost: ブースティングの回数
                model_param: modelのハイパーパラメータ
            """
            from sklearn.tree import DecisionTreeRegressor
            self.boost = boost
            self.model = DecisionTreeRegressor if model=='default' else model
            self.model_param = model_param
            self.trees = None
            self.beta = None

        def fit(self, x, y):
            """ブースティングを用いてモデルを学習させる
            
            Args:
                x: 学習データ(pd.DataFrame)
                y: 正解データ(pd.DataFrame)
            """
            x, y = np.array(x), np.array(y).reshape((-1, 1))
            # ブースティングで使用する変数
            _x, _y = x, y  # 引数を待避しておく
            self.trees = []  # 各機械学習モデルの配列
            self.beta = np.zeros((self.boost,))
            # 学習データに対する重み
            weights = np.ones((len(x),)) / len(x)
            # ブースティング
            for i in range(self.boost):
                # 決定木モデルを作成
                tree = self.model(**self.model_param)
                # 重み付きの機械学習モデルを代替するため、重みを確率にしてインデックスを取り出す
                all_idx = np.arange(x.shape[0])  # 全てのデータのインデックス
                p_weight = weights / weights.sum()  # 取り出す確率
                idx = np.random.choice(all_idx, size=x.shape[0], replace=True, p=p_weight)
                # インデックスの位置から学習用データを作成する
                x = _x[idx]
                y = _y[idx]
                # モデルを学習する
                tree.fit(x, y)
                # 一度、学習データに対して実行する
                z = tree.predict(x).reshape((-1, 1))
                # 差分の絶対値
                l = np.absolute(z - y).reshape((-1,))
                den = np.max(l)
                if den > 0:
                    loss = l / den  # 最大の差が1になるようにする
                err = np.sum(weights * loss) # ランダムな残差だと期待値が1/2になる
                print(f'iter #{i+1} -- error={err}')
                # 終了条件
                if i == 0 and err == 0:  # 最初に完全に学習してしまった
                    self.trees.append(tree)  # 最初のモデルだけ
                    self.beta = np.array([1])
                    break
                if err >= 0.5 or err == 0:
                    # 1/2より小さければ、判断しやすいデータとしにくいデータに傾向があるという事
                    self.beta = self.beta[:i]
                    break
                self.trees.append(tree)
                # AdaBoost.R2の計算
                self.beta[i] = err / (1.0 - err)
                weights *= [np.power(self.beta[i], 1.0 - lo) for lo in loss]
                weights /= weights.sum() # 重みの正規化

        def predict(self, x):
            """学習したモデルから予測する
            
            Args:
                x: 予測データ(pd.DataFrame)
            """
            x = np.array(x)
            # 各モデルの貢献度を求める
            w = np.log(1.0 / self.beta)
            if w.sum() == 0:
                w = np.ones((len(self.trees),)) / len(self.trees)
                print('w:', w)
            # 各モデルの実行結果を予め求めておく
            pred = [tree.predict(x).reshape((-1,)) for tree in self.trees]
            pred = np.array(pred).T  # 対角にするので(データの個数×モデルの数)になる
            # まずそれぞれのモデルの出力を、小さい順に並べて累積和を取る
            idx = np.argsort(pred, axis=1)  # 小さい順番の並び
            cdf = w[idx].cumsum(axis=1)  # 貢献度を並び順に累積してゆく
            cbf_last = cdf[:,-1].reshape((-1,1))  # 累積和の最後から合計を取得して整形
            # 下界を求める〜プログラム上は全部計算する
            above = cdf >= (1 / 2) * cbf_last   # これはTrueとFalseの二次元配列になる
            # 下界となる場所のインデックスを探す
            median_idx = above.argmax(axis=1)   # Trueが最初に現れる位置
            # そのインデックスにある出力の場所（最初に並べ替えたから）
            median_estimators = idx[np.arange(len(x)), median_idx]
            # その出力の場所にある実行結果の値を求めて返す
            result = pred[np.arange(len(x)), median_estimators]
            return result.reshape((-1, 1))  # 元の次元の形に戻す

        def __str__(self):
            """学習結果の可視化"""
            s = []
            w = np.log(1.0 / self.beta)
            if w.sum() == 0:
                w = np.ones((len(self.trees),)) / len(self.trees)
            else:
                w /= w.sum()
            for i, t in enumerate(self.trees):
                s.append(f'tree: #{i+1} -- weight={w[i]}')
                s.append(str(t))
            return '\n'.join(s)
        
