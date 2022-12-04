class NN:
    """ニューラルネットワーク(多層パーセプトロン)"""
    
    def __init__(self, hidden_neurons, epoch=1000, eta=0.01):
        """ニューラルネットワークの初期化を行う
        
        Args:
            hidden_neurons: 隠れ層のニューロン数(int)
            epoch: 繰り返し回数(default:1000)(int)
            eta: 学習率(default:0.001)(float)
        """
        self.hneurons = hidden_neurons
        self.epoch = epoch
        self.eta = eta

    def weight_initializer(self):
        # 隠れ層の重みとバイアスを初期化
        self.w1 = np.random.normal(
            0.0,                       # 平均は0
            pow(self.inneurons, -0.5), # 標準偏差は入力層のニューロン数を元に計算
            (self.hneurons,            # 行数は隠れ層のニューロン数
             self.inneurons + 1)       # 列数は入力層のニューロン数 + 1
            )
        
       # 出力層の重みとバイアスを初期化
        self.w2 = np.random.normal(
            0.0,                      # 平均は0
            pow(self.hneurons, -0.5), # 標準偏差は隠れ層のニューロン数を元に計算
            (self.oneurons,           # 行数は出力層のニューロン数
             self.hneurons + 1)       # 列数は隠れ層のニューロン数 + 1
            )
    
    def create_onehot_matrix(self, Y):
        """正解データを要素毎にOheHotEncodingする
        
        Args:
            Y: 正解データ(pd.DataFrame)
        """
        Y_get_dummies = pd.get_dummies(Y[Y.columns[0]])
        self.target_columns = Y_get_dummies.columns
        return Y_get_dummies
    
    def sigmoid(self, x):
        """シグモイド関数
        
        Args:
            x: 関数を適用するデータ
        """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """ソフトマックス関数
        
        Args:
            x: 関数を適用するデータ
        """
        c = np.max(x)
        exp_x = np.exp(x - c) # オーバーフローを防止する
        sum_exp_x = np.sum(exp_x)
        y = exp_x / sum_exp_x
        return y
    
    def fit(self, X, Y):
        """ニューラルネットワークの学習を行う
        
        Args:
            X: 訓練データ(pd.DataFrame)
            Y: 正解ラベル(pd.DataFrame)
        """
        print(f"ニューラルネットワークによる学習を開始します(epoch={self.epoch}、eta={self.eta})")
        self.X = X
        self.Y = Y
        self.inneurons = len(X.iloc[0])
        self.oneurons = len(Y[Y.columns[0]].unique())
        self.weight_initializer()
        Y = self.create_onehot_matrix(Y)
        
        # epochの回数だけ学習を繰り返す
        for eph in range(1, self.epoch+1):
            for inputs, targets in zip(np.array(X), np.array(Y)):
                ## [入力層]
                # 入力値の配列にバイアス項を追加して入力層から出力する
                inputs = np.array(
                    np.append(inputs, [1]), # 配列の末尾にバイアスのための「1」を追加
                    ndmin=2                # 2次元化
                ).T                        # 転置して1列の行列にする

                ## [隠れ層]
                # 入力層の出力に重み、バイアスを適用して隠れ層に入力する
                hidden_inputs = np.dot(
                    self.w1,              # 隠れ層の重み
                    inputs                # 入力層の出力
                    )
                # シグモイド関数を適用して隠れ層から出力
                hidden_outputs = self.sigmoid(hidden_inputs)        
                # 隠れ層の出力行列の末尾にバイアスのための「1」を追加
                hidden_outputs = np.append(
                    hidden_outputs,      # 隠れ層の出力行列
                    [[1]],               # 2次元形式でバイアス値を追加
                    axis=0               # 行を指定(列は1)
                    )

                ## [出力層]
                # 出力層への入力信号を計算
                final_inputs = np.dot(
                    self.w2,             # 隠れ層と出力層の間の重み
                    hidden_outputs       # 隠れ層の出力
                    )
                # ソフトマックス関数を適用して出力層から出力する
                final_outputs = self.softmax(final_inputs)

                ## ---バックプロパゲーション---(出力層)
                # 正解ラベルの配列を1列の行列に変換する
                targets = np.array(
                    targets,             # 正解ラベルの配列
                    ndmin=2              # 2次元化
                    ).T                  # 転置して1列の行列にする
                # 出力値と正解ラベルとの誤差
                output_errors = final_outputs - targets
                # 出力層の入力誤差δを求める
                delta_output = output_errors*(1 - final_outputs)*final_outputs
                # 重みを更新する前に隠れ層の出力誤差を求めておく
                hidden_errors = np.dot(
                    self.w2.T,           # 出力層の重み行列を転置する
                    delta_output         # 出力層の入力誤差δ
                    )
                # 出力層の重み、バイアスの更新
                self.w2 -= self.eta * np.dot(
                    # 出力誤差＊(1－出力信号)＊出力信号 
                    delta_output,
                    # 隠れ層の出力行列を転置
                    hidden_outputs.T
                )

                ## ---バックプロパゲーション---(隠れ層)
                # 逆伝搬された隠れ層の出力誤差からバイアスのものを取り除く
                hidden_errors_nobias = np.delete(
                    hidden_errors,      # 隠れ層のエラーの行列
                    self.hneurons,      # 隠れ層のニューロン数をインデックスにする
                    axis=0              # 行の削除を指定
                    )
                # 隠れ層の出力行列からバイアスを除く
                hidden_outputs_nobias = np.delete(
                    hidden_outputs,     # 隠れ層の出力の行列
                    self.hneurons,      # 隠れ層のニューロン数をインデックスにする
                    axis=0              # 行の削除を指定
                    )
                # 隠れ層の重み、バイアスの更新
                self.w1 -= self.eta * np.dot(
                    # 逆伝搬された隠れ層の出力誤差＊(1－隠れ層の出力)＊隠れ層の出力 
                    hidden_errors_nobias*(1.0 - hidden_outputs_nobias
                                         )*hidden_outputs_nobias,
                    # 入力層の出力信号の行列を転置
                    inputs.T
                    )
                
            # epochの初回・100回毎に分類器の精度を照会する
            if eph==1 or eph%10==0:
                sumA = sum([(self.predict(self.X)['prediction'].iloc[i] == self.Y.reset_index(drop=True).iloc[i])[0] for i in range(len(self.Y))])
                accuracy = sumA / len(X)
                print(f"epoch: {eph}/{self.epoch}  accuracy: {accuracy}")
                if accuracy == 1.0: break

    def predict(self, x):
        """テストデータを学習した重みで予測する
        
        Args:
            x: テスト用データの配列(pd.DataFrame)
        """
        final_outputs_list = []
        
        for inputs in np.array(x):
            ## [入力層]
            # 入力値の配列にバイアス項を追加して入力層から出力する
            inputs = np.array(
                np.append(inputs, [1]),      # 配列の末尾にバイアスの値「1」を追加
                ndmin=2                      # 2次元化
            ).T                              # 転置して1列の行列にする

            ## [隠れ層]
            # 入力層の出力に重み、バイアスを適用して隠れ層に入力する
            hidden_inputs = np.dot(self.w1,  # 入力層と隠れ層の間の重み
                                   inputs    # テストデータの行列
                                  )       
            # 活性化関数を適用して隠れ層から出力する
            hidden_outputs = self.sigmoid(hidden_inputs)

            ## [出力層]
            # 出力層への入力信号を計算
            final_inputs = np.dot(
                self.w2,                        # 隠れ層と出力層の間の重み
                np.append(hidden_outputs, [1]), # 隠れ層の出力配列の末尾にバイアスの値「1」を追加
                )       
            # 活性化関数を適用して出力層から出力する
            final_outputs = self.softmax(final_inputs)
            final_outputs_list.append(final_outputs)
        
        # 出力層からの出力を戻り値として返す
        result = pd.DataFrame(final_outputs_list, columns=self.target_columns)
        result['prediction'] = result.apply(lambda x : x[x==max(x)].index[0], axis=1)
        return result
