# テストデータ
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_boston, load_diabetes

def generate_data(load):
    """sklearn.datasetsのデータを読み込む"""
    data = pd.DataFrame(load.data, columns=load.feature_names)
    target = pd.DataFrame(load.target, columns=["target"])
    #return train_test_split(data, target, test_size=0.2, random_state=71)
    return pd.concat([data, target], axis=1)

df_iris = generate_data(load_iris())
df_wine = generate_data(load_wine())
df_boston = generate_data(load_boston())
df_diabetes = generate_data(load_diabetes())


# Decision Tree テスト
np.random.seed(1)
df = df_boston
x = df[df.columns[:-1]]
max_depth = 3
is_regression = True

if not is_regression:
    print("分類")
    y = df[df.columns[-1]]
    mt = 'gini'
    lf = LogisticRegression
    plf = DecisionTree(prunfnc='critical', pruntest=False, splitratio=0.5, critical=0.8,
            max_depth=max_depth, metric=mt, leaf=lf, depth=1, is_regression=is_regression)
    plf.fit(x,y)
    z = plf.predict(x)
    print(str(plf))
    print(z)
else:
    print("回帰")
    y = df[df.columns[-1]]
    mt = 'dev'
    lf = LinearRegression
    plf = DecisionTree(prunfnc='critical', pruntest=False, splitratio=0.5, critical=0.8,
            max_depth=max_depth, metric=mt, leaf=lf, depth=1, is_regression=is_regression)
    plf.fit(x,y)
    z = plf.predict(x)
    print(str(plf))
    print(z)
