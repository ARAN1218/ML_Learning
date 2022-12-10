import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_boston

# シミュレーションデータの生成(分類)
def generate_data_classification():
    heacet = load_iris()
    heacet_data = pd.DataFrame(heacet.data, columns=["がく片の長さ","がく片の幅","花びらの長さ","花びらの幅"])
    heacet_target = pd.DataFrame(heacet.target, columns=["花の種類"])
    # heacet_all = pd.concat([heacet_data,heacet_target], axis=1)
    return train_test_split(heacet_data, heacet_target, test_size=0.2, random_state=42)

X_train1, X_test1, y_train1, y_test1 = generate_data_classification()
y_train1['花の種類'] = y_train1['花の種類'].map(lambda x : 'タイプA' if x==0 else 'タイプB')
y_test1['花の種類'] = y_test1['花の種類'].map(lambda x : 'タイプA' if x==0 else 'タイプB')

# テスト(分類)
knn = KNN(K=3, types="classification")
knn.fit(X_train1, y_train1)
for method in ("Euclid", "Manhattan", "Mahalanobis", "Chebyshev"):
    print(method)
    test = knn.predict(X_test1.iloc[:10], method=method)
    display(pd.concat([y_test1.iloc[:10].reset_index(drop=True), pd.DataFrame(test, columns=['prediction'])], axis=1))



# シミュレーションデータの生成(回帰)
def generate_data_regression():
    boston = load_boston()
    boston_data = pd.DataFrame(boston.data, columns=boston.feature_names)
    boston_target = pd.DataFrame(boston.target, columns=['MEDV'])
    # boston_all = pd.concat([boston_data,boston_target], axis=1)
    return train_test_split(boston_data, boston_target, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = generate_data_regression()

# テスト(回帰)
knn = KNN(K=3, types="regression")
knn.fit(X_train, y_train)
for method in ("Euclid", "Manhattan", "Mahalanobis", "Chebyshev"):
    print(method)
    test = knn.predict(X_test.iloc[:10], method=method)
    display(pd.concat([y_test.iloc[:10].reset_index(drop=True), pd.DataFrame(test, columns=['prediction'])], axis=1))
