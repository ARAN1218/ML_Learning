import numpy as np
import pandas as pd


# 自作Naive_Bayes
# 離散値は確率、連続値は確率密度を計算する
class Naive_bayes:
    def __init__(self, Laplace=1):
        self.Laplace = Laplace # ラプラス平滑化のハイパーパラメータ
        self.proba_dict = {} # 推定した各属性の条件付き確率を保持する
        
    def _discrete(self, X, y, column):
        self.proba_dict[column] = pd.crosstab(X[column], y[self.y_column])
        for y_val in self.y_values: self.proba_dict[column][y_val] = (self.proba_dict[column][y_val]+self.Laplace) / (self.y_value_counts[y_val]+len(self.proba_dict[column].index))
    
    def _continuous(self, X, y, column):
        self.proba_dict[column] = pd.DataFrame([{"mean":X[y[self.y_column]==y_val][column].mean(), "var":X[y[self.y_column]==y_val][column].var()} for y_val in self.y_values], index=self.y_values)
        self.proba_dict[column] = self.proba_dict[column] + self.Laplace #.apply(lambda x : x+((self.Laplace-x["mean"])/(x["var"]+self.Laplace)), axis=1) # 平均・分散が０になることを防ぐため、Laplaceを標準化した値を足す
    
    def fit(self, X, y):
        # 目的変数の情報を保持しておく
        self.y_column = y.columns[0]
        self.y_values = y[self.y_column].unique()
        self.y_value_counts = y[self.y_column].value_counts()
        
        # クラス事前確率を推定する(ラプラス平滑化)
        #self.proba_dict[self.y_column] = (y.value_counts() + self.Laplace) / (len(y) + len(self.y_values))
        
        # 各属性の条件付き確率を推定する
        for column in X.columns:
            if X[column].dtype in ("str", "object"): # 説明変数が離散値(str or object)
                self._discrete(X, y, column)
            else: # 説明変数が連続値(str以外)
                self._continuous(X, y, column)
    
    def _discrete_proba(self, X, column):
        # nanはラプラス平滑化を加味して1/(データ数+クラス数)でfillnaしたい
        for y_val in self.y_values:
            self.predict_proba[y_val] = pd.merge(self.predict_proba[y_val], self.proba_dict[column].reset_index()[[column, y_val]], on=column, how="left")
            self.predict_proba[y_val].iloc[:,-1] = self.predict_proba[y_val].iloc[:,-1].fillna(1/(self.y_value_counts[y_val]+len(self.proba_dict[column].index)))
    
    def _continuous_proba(self, X, column):
        for y_val in self.y_values: self.predict_proba[y_val][column+"_proba"] = self.predict_proba[y_val][column].map(lambda x : (1/(2*np.pi*self.proba_dict[column]["var"][y_val])**0.5) * np.exp(-1*((x-self.proba_dict[column]["mean"][y_val])**2 / (2*self.proba_dict[column]["var"][y_val]))))
    
    def predict(self, X):
        # 各属性の条件付き確率を推定する
        # 離散値と連続値で確率の求め方が違うことに注意する
        self.predict_proba = dict([(y_val, X.copy()) for y_val in self.y_values])
        for column in X.columns:
            if X[column].dtype in ("str", "object"): # 説明変数が離散値(str or object)
                self._discrete_proba(X, column)
            else: # 説明変数が連続値(str以外)
                self._continuous_proba(X, column)
    
        # データ毎に予測確率を計算する
        self.prediction = pd.DataFrame([self.predict_proba[i].iloc[:,len(self.proba_dict):].apply(np.prod, axis=1) for i in self.predict_proba.keys()]).T
        return self.prediction.apply(lambda x : self.y_values[np.argmax(x)], axis=1).reset_index(drop=True)
    
    def score(self, X, y):
        # 正解率で評価する
        pred = self.predict(X)
        df_score = pd.concat([pred, y.reset_index(drop=True)], axis=1).set_axis(["pred", "true"], axis=1)
        df_score["accuracy"] = (df_score["pred"] == df_score["true"])
        return sum(df_score["accuracy"]) / len(df_score)
