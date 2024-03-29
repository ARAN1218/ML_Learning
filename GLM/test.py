# 重回帰分析　テスト
## 最小二乗法
# シミュレーションデータの生成(回帰)
def generate_data_regression():
    boston = load_boston()
    boston_data = pd.DataFrame(boston.data, columns=boston.feature_names)
    boston_target = pd.DataFrame(boston.target, columns=['MEDV'])
    # boston_all = pd.concat([boston_data,boston_target], axis=1)
    return train_test_split(boston_data, boston_target, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = generate_data_regression()

mra = MRA()
mra.fit(X_train, y_train)
mra.predict(X_test)


# 二項ロジスティック回帰　テスト
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# シミュレーションデータの生成(2クラス分類)
def generate_data_classification():
    heacet = load_iris()
    heacet_data = pd.DataFrame(heacet.data, columns=["がく片の長さ","がく片の幅","花びらの長さ","花びらの幅"])
    heacet_target = pd.DataFrame(heacet.target, columns=["花の種類"])
    heacet_all = pd.concat([heacet_data,heacet_target], axis=1).query("花の種類 != 2")
    return train_test_split(heacet_all.iloc[:,:4], heacet_all[['花の種類']], test_size=0.2, random_state=42)

X_train2, X_test2, y_train2, y_test2 = generate_data_classification()

lr = LogisticRegression(eta=0.00001, epoch=2000)
lr.fit(X_train2, y_train2)
pd.concat([y_test2.reset_index(drop=True), pd.DataFrame(lr.predict(X_test2, threshold=0.5), columns=['pred'])], axis=1)


# 順序ロジスティック回帰　テスト
# シミュレーションデータの生成(3クラス分類(順序あり))
def generate_data_classification3_ordinal():
    A = pd.DataFrame(np.random.randint(0,100,(100,1)), columns=['X'])
    B = pd.DataFrame(np.random.randint(50,150,(100,1)), columns=['X'])
    C = pd.DataFrame(np.random.randint(100,200,(100,1)), columns=['X'])
    #D = pd.DataFrame(np.random.randint(301,400,(100,1)), columns=['X'])
    A['Y'], B['Y'], C['Y'] = 0, 1, 2
    df = pd.concat([A,B,C], axis=0)
    display(df)
    return train_test_split(df[['X']], df[['Y']], test_size=0.2, random_state=42)

X_train3_o, X_test3_o, y_train3_o, y_test3_o = generate_data_classification3_ordinal()

lr = OLR(eta=0.00001, epoch=2000)
lr.fit(X_train3_o, y_train3_o)
pd.concat([y_test3_o.reset_index(drop=True), pd.DataFrame(lr.predict(X_test3_o, threshold=0.5), columns=['pred'])], axis=1)


# 多項ロジスティック回帰　テスト
# シミュレーションデータの生成(3クラス分類)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def generate_data_classification3():
    heacet = load_iris()
    heacet_data = pd.DataFrame(heacet.data, columns=["がく片の長さ","がく片の幅","花びらの長さ","花びらの幅"])
    heacet_target = pd.DataFrame(heacet.target, columns=["花の種類"])
    heacet_all = pd.concat([heacet_data,heacet_target], axis=1)
    return train_test_split(heacet_all.iloc[:,:4], heacet_all[['花の種類']], test_size=0.2, random_state=42)

X_train3, X_test3, y_train3, y_test3 = generate_data_classification3()

lr = MLR(eta=0.001, epoch=2000)
lr.fit(X_train3, y_train3)
pd.concat([y_test3.reset_index(drop=True), pd.DataFrame(lr.predict(X_test3))], axis=1)


# ポアソン回帰　テスト
# シミュレーションデータの生成(ポアソン回帰)
def generate_data_poisson():
    data = pd.DataFrame(np.random.poisson(lam=10, size=1000), columns=['Y'])
    data['X1'] = data['Y'].map(lambda x : x + np.random.randint(-20,2))#data['Y'].map(lambda x : 1 if x>=10 else 0)
    data['X2'] = data['Y'].map(lambda x : x + np.random.randint(-2,2))
    return train_test_split(data[['X1', 'X2']], data[['Y']], test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = generate_data_poisson()

# テスト
lr = PoissonRegression(eta=0.00001, epoch=1000)
lr.fit(X_train, y_train)
pd.concat([y_test.reset_index(drop=True), pd.DataFrame(lr.predict(X_test), columns=['pred'])], axis=1)
