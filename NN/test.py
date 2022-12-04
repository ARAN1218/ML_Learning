import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# シミュレーションデータの生成
def generate_data():
    heacet = load_iris()
    heacet_data = pd.DataFrame(heacet.data, columns=["がく片の長さ","がく片の幅","花びらの長さ","花びらの幅"])
    heacet_target = pd.DataFrame(heacet.target, columns=["花の種類"])
    heacet_all = pd.concat([heacet_data,heacet_target], axis=1).query("花の種類 != 2")
    return train_test_split(heacet_all.iloc[:, :4], heacet_all.iloc[:, 4:], test_size=0.2, random_state=42)


# 単純パーセプトロン テスト  
X_train, X_test, y_train, y_test = generate_data()
perceptron = PERCEPTRON(epoch=100, eta=0.001)
perceptron.fit(X_train, y_train)
pd.concat([perceptron.predict(X_test), y_test], axis=1)



# ニューラルネットワーク(多層パーセプトロン) テスト 
X_train, X_test, y_train, y_test = generate_data()
y_train['花の種類'] = y_train['花の種類'].map(lambda x : 'タイプA' if x==0 else 'タイプB')
y_test['花の種類'] = y_test['花の種類'].map(lambda x : 'タイプA' if x==0 else 'タイプB')

h_neuronss = 3 # 隠れ層のニューロンの数
loop = 1000 # 繰り返し回数
eta = 0.001 # 学習率

# 学習
nn = NN(h_neuronss, loop, eta)
nn.fit(X_train, y_train)
# 予測
pred = nn.predict(X_test)
pd.concat([pred, y_test.reset_index(drop=True)], axis=1)
