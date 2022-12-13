class DecisionTree:
    """決定木アルゴリズム"""
    
    def __init__(self, prunfnc='critical', pruntest=False, splitratio=0.5, critical=0.8,
                max_depth=5, metric='gini', leaf='zeror', depth=1, is_regression=False):
        """初期化
        
        Args:
            metric: 損失関数
                dev: 標準偏差
                gini: ジニ不純度
                infgain: Information gain
            leaf: 葉に仕込む機械学習アルゴリズム
            max_depth: 決定木の深さ
            depth: 現在の木の階層(再帰用)
            prunfnc: プルーニング用関数
            pruntest: プルーニング用にテスト用データを取り分けるか
            splitratio: プルーニング用テストデータの割合
            critical: "critical"プルーニング用の閾値
            is_regression: 分類 or 回帰
        """
        self.metric = self.Entropy.deviation if is_regression or metric=='dev' else \
                    self.Entropy.gini if metric=='gini' else \
                    self.Entropy.infgain
        self.leaf = ZeroRule if leaf=='zeror' else leaf
        self.max_depth = max_depth
        self.depth = depth
        self.prunfnc = prunfnc
        self.pruntest = pruntest
        self.splitratio = splitratio
        self.critical = critical
        self.is_regression = is_regression
        
    class Entropy:
        """決定木の損失関数"""

        def deviation(self, y):
            """標準偏差"""
            return y.std()

        def gini(self, y):
            """ジニ不純度"""
            m = y.sum(axis=0)
            size = y.shape[0]
            e = [(p / size) ** 2 for p in m]
            return 1.0 - np.sum(e)

        def infgain(self, y):
            """Information gain"""
            m = y.sum(axis=0)
            size = y.shape[0]
            e = [p * np.log2(p / size) / size for p in m if p != 0.0]
            return -np.sum(e)
        
    class ZeroRule:
        """単純な予測アルゴリズム
        
        分類: 多数決投票
        回帰: 目的変数の平均値
        """
        def __init__(self):
            self.r = None

        def fit(self, x, y):
            self.r = np.mean(y, axis=0)
            return self

        def predict(self, x):
            z = np.zeros((len(x), self.r.shape[0]))
            return z + self.r

        def __str__(self):
            return str(self.r)
        
    def make_split(self, feat, val):
        """featをval以下と以上で分割するインデックスを返す
        
        Args:
            feat: ある特徴量(np.array)
            val: ある特徴量の分岐点に用いる値(float)
        """
        left, right = [], []
        for i, v in enumerate(feat):
            if v < val:
                left.append(i)
            else:
                right.append(i)
        return left, right

    def make_loss(self, y1, y2):
        """yをy1とy2で分割したときのMetrics関数の重み付き合計を返す
        
        Args:
            y1, y2: 左右に分割した目的変数(np.array)
        """
        if y1.shape[0] == 0 or y2.shape[0] == 0:
            return np.inf
        total = y1.shape[0] + y2.shape[0]
        m1 = self.metric(self,y1) * (y1.shape[0] / total)
        m2 = self.metric(self,y2) * (y2.shape[0] / total)
        return m1 + m2

    def get_node(self):
        """新しく決定木ノードを作成する(再帰処理)"""
        return DecisionTree(prunfnc=self.prunfnc, pruntest=self.pruntest,
                            max_depth=self.max_depth, metric=self.metric, 
                            splitratio=0.5, critical=0.8, 
                            leaf=self.leaf, depth=self.depth + 1)
    
    def reducederror(self, node, x, y):
        """決定木のプルーニング(枝刈り)を行う(Reduce Error用)
        
        Args:
            node: この決定木のノード
            x: プルーニング判定用の学習データ(np.array)
            y: プルーニング判定用の正解データ(np.array)
        """
        # ノードが葉でなかったら
        if isinstance(node, DecisionTree):
            # 左右の分割を得る
            feat = x[:,node.feat_index]
            val = node.feat_val
            l, r = node.make_split(feat, val)
            # 左左右にデータが振り分けられるか
            if val is np.inf or len(r) == 0:
                return reducederror(node.left, x, y) # 一つの枝のみの場合、その枝で置き換える
            elif len(l) == 0:
                return reducederror(node.right, x, y) # 一つの枝のみの場合、その枝で置き換える
            # 左右の枝を更新する
            node.left = reducederror(node.left, x[l], y[l])
            node.right = reducederror(node.right, x[r], y[r])
            # 学習データに対するスコアを計算する
            p1 = node.predict(x)
            p2 = node.left.predict(x)
            p3 = node.right.predict(x)
            # クラス分類かどうか
            if self.is_regression==False: #y.shape[1] > 1:
                # 誤分類の個数をスコアにする
                ya = y.argmax(axis=1)
                d1 = np.sum(p1.argmax( axis=1 ) != ya)
                d2 = np.sum(p2.argmax( axis=1 ) != ya)
                d3 = np.sum(p3.argmax( axis=1 ) != ya)
            else:
                # 二乗平均誤差をスコアにする
                d1 = np.mean((p1 - y) ** 2)
                d2 = np.mean((p2 - y) ** 2)
                d3 = np.mean((p3 - y) ** 2)
            if d2 <= d1 or d3 <= d1: # 左右の枝どちらかだけでスコアが悪化しない
                # スコアの良い方の枝を返す
                if d2 < d3:
                    return node.left
                else:
                    return node.right
        # 現在のノードを返す
        return node

    def getscore(self, node, score):
        """全てのノードのスコアを取得する(Critical value用)
        
        Args:
            node: 確認するノード
            score: 計算したスコアを保存しておくリスト(list[float])
        """
        # ノードが葉でなかったら
        if isinstance(node, DecisionTree):
            if node.score >= 0 and node.score is not np.inf:
                score.append(node.score)
            self.getscore(node.left, score)
            self.getscore(node.right, score)

    def criticalscore(self, node, score_max):
        """決定木のプルーニング(枝刈り)を行う(Reduce Error用)
        
        Args:
            node: 確認するノード
            score_max: 最高のスコア
        """
        # ノードが葉でなかったら
        if isinstance(node, DecisionTree):
            # 左右の枝を更新する
            node.left = self.criticalscore(node.left, score_max)
            node.right = self.criticalscore(node.right, score_max)
            # ノードを削除
            if node.score > score_max:
                leftisleaf = not isinstance(node.left, DecisionTree)
                rightisleaf = not isinstance(node.right, DecisionTree)
                # 両方共葉ならば一つの葉にする
                if leftisleaf and rightisleaf:
                    return node.left
                # どちらかが枝ならば枝の方を残す
                elif leftisleaf and not rightisleaf:
                    return node.right
                elif not leftisleaf and rightisleaf:
                    return node.left
                # どちらも枝ならばスコアの良い方を残す
                elif node.left.score < node.right.score:
                    return node.left
                else:
                    return node.right
        # 現在のノードを返す
        return node

    def fit_leaf(self, x, y):
        """既に作成されている枝に従ってデータを分割し、現在のノードが葉であればfitメソッドを実行する
        
        Args:
            x: 学習データ(np.array)
            y: 正解データ(np.array)
        """
        # 説明変数から分割した左右のインデックスを取得
        feat = x[:,self.feat_index]
        val = self.feat_val
        l, r = self.make_split(feat, val)
        # 葉のみを学習させる
        if len(l) > 0:
            if isinstance(self.left, DecisionTree):
                self.left.fit_leaf(x[l], y[l])
            else:
                try:
                    self.left.fit(x[l], y[l]) # np.argmax(y[l], axis=1)
                except:
                    self.left = self.ZeroRule()
                    self.left.fit(x[l], y[l])
        if len(r) > 0:
            if isinstance(self.right, DecisionTree):
                self.right.fit_leaf(x[r], y[r])
            else:
                try:
                    self.right.fit(x[r], y[r]) # np.argmax(y[r], axis=1)
                except:
                    self.right = self.ZeroRule()
                    self.right.fit(x[r], y[r])

    def split_tree(self, x, y):
        """決定木を分割する
        
        Args:
            x: 学習データ(pd.DataFrame)
            y: 正解データ(pd.DataFrame)
        """
        # データを分割して左右の枝に属するインデックスを返す
        self.feat_index = 0
        self.feat_val = np.inf
        score = np.inf
        # データの前準備
        ytil = y[:,np.newaxis]
        #print(ytil)
        xindex = np.argsort(x, axis=0)
        #print(xindex)
        ysot = np.take(ytil, xindex, axis=0)
        #print(ysot)
        for f in range(x.shape[0]):
            # 小さい方からf個の位置にある値で分割
            l = xindex[:f,:]
            r = xindex[f:,:]
            #print(ysot)
            ly = ysot[:f, :, 0, :]
            ry = ysot[f:, :, 0, :]
            # 全ての次元のスコアを求める
            loss = [self.make_loss(ly[:, yp, :], ry[:, yp, :])
                    if x[xindex[f-1,yp], yp] != x[xindex[f,yp], yp] else np.inf
                    for yp in range(x.shape[1])]
            # 最小のスコアになる次元
            i = np.argmin(loss)
            if score > loss[i]:
                score = loss[i]
                self.feat_index = i
                self.feat_val = x[xindex[f,i], i]
        # 実際に分割するインデックスを取得
        _filter =  x[:, self.feat_index] < self.feat_val
        left = np.where(_filter)[0].tolist()
        right = np.where(_filter==False)[0].tolist()
        self.score = score
        return left, right

    def fit(self, x, y):
        """決定木の学習の一連の流れを行う
        
        Args:
            x: 学習データ(pd.DataFrame)
            y: 正解データ(pd.DataFrame)
        """
        x, y = np.array(x), np.array(y).reshape((-1, 1))
        # 深さ＝１，根のノードの時のみ
        if self.depth == 1 and self.prunfnc is not None:
            # プルーニングに使うデータ
            x_t, y_t = x, y
            # プルーニング用にテスト用データを取り分けるならば
            if self.pruntest:
                # 学習データとテスト用データを別にする
                n_test = int(round(len(x) * self.splitratio))
                n_idx = np.random.permutation(len(x))
                tmpx = x[n_idx[n_test:]]
                tmpy = y[n_idx[ n_test:]]
                x_t = x[n_idx[:n_test]]
                y_t = y[n_idx[:n_test]]
                x = tmpx
                y = tmpy

        # 決定木の学習・・・"critical"プルーニング時は木の分割のみ
        self.left = self.leaf()
        self.right = self.leaf()
        left, right = self.split_tree(x, y)
        if self.depth < self.max_depth:
            if len(left) > 0:
                self.left = self.get_node()
            if len(right) > 0:
                self.right = self.get_node()
        if self.depth < self.max_depth or self.prunfnc != 'critical':
            if len(left) > 0:
                #print(x[left])
                #print(y[left])
                #print(np.argmax(y[left], axis=1))
                self.left.fit(x[left], y[left]) # np.argmax(y[left], axis=1)
            if len(right) > 0:
                self.right.fit(x[right], y[right]) # np.argmax(y[right], axis=1)

        # 深さ＝１，根のノードの時のみ
        if self.depth == 1 and self.prunfnc is not None:
            if self.prunfnc == 'reduce':
                # プルーニングを行う
                self.reducederror(self, x_t, y_t)
            elif self.prunfnc == 'critical':
                # 学習時のMetrics関数のスコアを取得する
                score = []
                self.getscore(self, score)
                if len(score) > 0:
                    # スコアから残す枝の最大スコアを計算
                    i = int(round(len(score) * self.critical))
                    score_max = sorted(score)[min(i, len(score)-1)]
                    # プルーニングを行う
                    self.criticalscore(self, score_max)
                # 葉を学習させる
                self.fit_leaf(x, y)

    def predict(self, x):
        """学習した重みを用いて予測
        
        Args:
            x: テストデータ(pd.DataFrame)
        """
        x = np.array(x)
        # 説明変数から分割した左右のインデックスを取得
        feat = x[:,self.feat_index]
        val = self.feat_val
        l, r = self.make_split(feat, val)
        # 左右の葉を実行して結果を作成する
        z = None
        if len(l) > 0 and len(r) > 0:
            left = self.left.predict(x[l]).reshape((-1, 1))
            right = self.right.predict(x[r]).reshape((-1, 1))
            z = np.zeros((x.shape[0], left.shape[1]))
            z[l] = left
            z[r] = right
        elif len(l) > 0:
            z = self.left.predict(x)
        elif len(r) > 0:
            z = self.right.predict(x)
        return z
    
    def __str__(self):
        """str関数を適用することで決定木の学習内容を可視化"""
        return '\n'.join([
            f'  if feat[{self.feat_index}] <= {self.feat_val} then:',
            f'    {self.left}',
            '  else',
            f'    {self.right}'])
