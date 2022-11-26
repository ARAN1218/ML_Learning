# 各種ライブラリ
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib  # matplotlibの日本語表示対応
from scipy.special import factorial
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

np.random.seed(42)
pd.options.display.float_format = "{:.2f}".format
warnings.simplefilter("ignore")  # warningsを非表示に


# シミュレーションデータ生成関数
def generate_simulation_data(N, beta, mu, Sigma):
    """線形のシミュレーションデータを生成し、訓練データとテストデータに分割する
    
    Args: 
        N: インスタンスの数
        beta: 各特徴量の傾き
        mu: 各特徴量は多変量正規分布から生成される。その平均。
        Sigma: 各特徴量は多変量正規分布から生成される。その分散共分散行列。
    """

    # 多変量正規分布からデータを生成する
    X = np.random.multivariate_normal(mu, Sigma, N)

    # ノイズは平均0標準偏差0.1(分散は0.01)で決め打ち
    epsilon = np.random.normal(0, 0.1, N)

    # 特徴量とノイズの線形和で目的変数を作成
    y = X @ beta + epsilon
    
    X = pd.DataFrame(X, columns=['X'+str(i) for i in range(len(Sigma))])
    y = pd.DataFrame(y, columns=['y'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)


# シミュレーションデータの設定
N = 1000
J = 3
mu = np.zeros(J)
Sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
beta = np.array([0, 1, 2])

# シミュレーションデータの生成
X_train, X_test, y_train, y_test = generate_simulation_data(N, beta, mu, Sigma)
