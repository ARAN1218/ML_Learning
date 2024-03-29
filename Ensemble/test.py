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

    
# Bagging テスト
random.seed(1)
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

df = df_boston
x = df[df.columns[:-1]]
is_regression = True

# ベンチマークとなるアルゴリズムと、アルゴリズムを実装したモデルの一覧
classify_models = [ 
    (LogisticRegression, {}),
    (SVC, {'random_state':1}), 
    (GaussianProcessClassifier, {'random_state':1}),
    (KNeighborsClassifier, {}), 
    (MLPClassifier, {'random_state':1}) 
]

regression_models = [
    (LinearRegression, {}),
    (SVR, {}),
    (GaussianProcessRegressor, {'normalize_y':True, 'alpha':1, 'random_state':1}),
    (KNeighborsRegressor, {}),
    (MLPRegressor, {'hidden_layer_sizes':(5), 'solver':'lbfgs', 'random_state':1})
]

if not is_regression:
    print('分類')
    y = df[df.columns[-1]]
    plf = Bagging(models=classify_models, ratio=1.0, is_regression=is_regression)
    plf.fit(x, y)
    print(str(plf))
    z = plf.predict(x)
    print(z)
    
    from sklearn.metrics import classification_report, f1_score, accuracy_score
    print("accuracy:", accuracy_score(y, z))
    print("f1_score:", f1_score(y, z, average='weighted'))
else:
    print('回帰')
    y = df[df.columns[-1]]
    plf = Bagging(models=regression_models, ratio=1.0, is_regression=is_regression)
    plf.fit(x, y)
    print(str(plf))
    z = plf.predict(x)
    print(z)

    from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
    print('Train Score:')
    rp = r2_score(y, z)
    print('R2 Score: %f'%rp)
    rp = explained_variance_score(y, z)
    print('Explained Variance Score: %f'%rp)
    rp = mean_absolute_error(y, z)
    print('Mean Absolute Error: %f'%rp)
    rp = mean_squared_error(y, z)
    print('Mean Squared Error: %f'%rp)

    
# RandomForest テスト
#random.seed( 1 )
import pandas as pd

df = df_diabetes
x = df[df.columns[:-1]]
is_regression = True

if not is_regression:
    y = df[df.columns[-1]]
    plf = RandomForest(max_features=5, n_trees=5, ratio=1.0, tree_params={'max_depth':10}, is_regression=False)
    plf.fit(x,y)
    print(str(plf))
    z = plf.predict(x)
    print(z)
    
    from sklearn.metrics import classification_report, f1_score, accuracy_score
    print("accuracy:", accuracy_score(y, z))
    print("f1_score:", f1_score(y, z, average='weighted'))
else:
    y = df[df.columns[-1]]
    plf = RandomForest(max_features=5, n_trees=5, ratio=1.0, tree_params={'max_depth':10}, is_regression=True)
    plf.fit(x,y)
    print(str(plf))
    z = plf.predict(x)
    print(z)
    
    from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error

    print('Train Score:')
    rp = r2_score(y, z)
    print('R2 Score: %f'%rp)
    rp = explained_variance_score(y, z)
    print('Explained Variance Score: %f'%rp)
    rp = mean_absolute_error(y, z)
    print('Mean Absolute Error: %f'%rp)
    rp = mean_squared_error( y, z )
    print('Mean Squared Error: %f'%rp)
    
    
# AdaBoost テスト
# テスト(分類)
import pandas as pd
from sklearn.linear_model import LogisticRegression
df = df_wine#.query("target != 2")
x = df[df.columns[:-1]]
y = df[df.columns[-1]]

plf = AdaBoost(boost=5, model_param={'max_depth':2}, adatype='M1') # モデル指定なし(default)
#plf = AdaBoost(model=LogisticRegression, boost=5, model_param={}, adatype='M1') # モデル指定あり

plf.fit(x, y)
print(str(plf))
z = plf.predict(x)

from sklearn.metrics import classification_report, f1_score, accuracy_score
print("accuracy:", accuracy_score(y, z))
print("f1_score:", f1_score(y, z, average='weighted'))


# テスト(回帰)
import pandas as pd
from sklearn.linear_model import LinearRegression
df = df_boston
x = df[df.columns[:-1]]
y = df[df.columns[-1]]

plf = AdaBoost(boost=5, model_param={'max_depth':5}, adatype='R2', threshold=0.01) # モデル指定なし(default)
#plf = AdaBoost(model=LinearRegression, boost=5, model_param={}, adatype='R2', threshold=0.01) # モデル指定あり(線形回帰)

plf.fit(x,y)
z = plf.predict(x)
print(str(plf))

from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
print('Train Score:')
rp = r2_score(y, z)
print('R2 Score: %f'%rp)
rp = explained_variance_score(y, z)
print('Explained Variance Score: %f'%rp)
rp = mean_absolute_error(y, z)
print('Mean Absolute Error: %f'%rp)
rp = mean_squared_error(y, z)
print('Mean Squared Error: %f'%rp)


# モデル選択法　テスト
## 交差検証 テスト
import pandas as pd
np.random.seed(71)

df = df_iris
x = df[df.columns[:-1]]
y = df[df.columns[-1]]
is_regression = False
# ベンチマークとなるアルゴリズムと、アルゴリズムを実装したモデルの一覧
models = [
    [SVC(random_state=1), GaussianProcessClassifier(random_state=1),
    KNeighborsClassifier(), MLPClassifier(random_state=1)], 
    [SVR(), GaussianProcessRegressor(normalize_y=True, alpha=1, random_state=1),
    KNeighborsRegressor(), MLPRegressor(hidden_layer_sizes=(5), solver='lbfgs', random_state=1)]
]

if not is_regression:
    print("分類")
    cv = CVSelect(is_regression, models[0], K=5)
    cv.fit(x,y)
    z = cv.predict(x)
    print(str(cv))
    display(pd.DataFrame([y,z.squeeze()], index=['target','pred']).T)
    print((z.squeeze() == y).astype(np.float32).mean())
else:
    print("回帰")
    cv = CVSelect(is_regression, models[1], K=5)
    cv.fit(x,y)
    z = cv.predict(x)
    print(str(cv))
    display(pd.DataFrame([y,z.squeeze()], index=['target','pred']).T)
    print((abs(z.squeeze()-y)).mean())


## 情報量基準　テスト
import pandas as pd
np.random.seed(71)

df = df_boston
x = df[df.columns[:-1]]
y = df[df.columns[-1]]
is_regression = True
# ベンチマークとなるアルゴリズムと、アルゴリズムを実装したモデルの一覧
models = [
    [SVC(random_state=1), GaussianProcessClassifier(random_state=1),
    KNeighborsClassifier(), MLPClassifier(random_state=1)], 
    [SVR(), GaussianProcessRegressor(normalize_y=True, alpha=1, random_state=1),
    KNeighborsRegressor(), MLPRegressor(hidden_layer_sizes=(5), solver='lbfgs', random_state=1)]
]
ic = "AIC"

if not is_regression:
    print("分類")
    ic = ICSelect(is_regression, models[0], K=5, ic=ic)
    ic.fit(x,y)
    z = ic.predict(x)
    print(str(bic))
    display(pd.DataFrame([y,z.squeeze()], index=['target','pred']).T)
    print((z.squeeze() == y).astype(np.float32).mean())
else:
    print("回帰")
    ic = ICSelect(is_regression, models[1], K=5, ic=ic)
    ic.fit(x,y)
    z = ic.predict(x)
    print(str(ic))
    display(pd.DataFrame([y,z.squeeze()], index=['target','pred']).T)
    print((abs(z.squeeze()-y)).mean())


## バンピングテスト
import pandas as pd
np.random.seed(71)

df = df_boston
x = df[df.columns[:-1]]
y = df[df.columns[-1]]
is_regression = True
# ベンチマークとなるアルゴリズムと、アルゴリズムを実装したモデルの一覧
models = [
    [SVC(random_state=1), GaussianProcessClassifier(random_state=1),
    KNeighborsClassifier(), MLPClassifier(random_state=1)], 
    [SVR(), GaussianProcessRegressor(normalize_y=True, alpha=1, random_state=1),
    KNeighborsRegressor(), MLPRegressor(hidden_layer_sizes=(5), solver='lbfgs', random_state=1)]
]

if not is_regression:
    print("分類")
    bumping = Bumping(is_regression, tree.DecisionTreeClassifier, K=5, n_trees=5, ratio=1.0, tree_params={'max_depth':2})
    bumping.fit(x,y)
    z = bumping.predict(x)
    print(str(bumping))
    display(pd.DataFrame([y,z.squeeze()], index=['target','pred']).T)
    print((z.squeeze() == y).astype(np.float32).mean())
else:
    print("回帰")
    bumping = Bumping(is_regression, tree.DecisionTreeRegressor, K=5, n_trees=5, ratio=1.0, tree_params={'max_depth':10})
    bumping.fit(x,y)
    z = bumping.predict(x)
    print(str(bumping))
    display(pd.DataFrame([y,z.squeeze()], index=['target','pred']).T)
    print((abs(z.squeeze()-y)).mean())















