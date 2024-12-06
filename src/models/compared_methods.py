# 標本平均と最尤法の実装
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

class compared_methods:
    def __init__(self, X:np.array, y:np.array, X_mu:list, X_sigma:list) -> None:
        """
        :param X: サンプリング後のX
        :param y: サンプリング後のy
        :param X_mu: 共変量の母集団分布の平均リスト
        :param X_sigma: 共変量の母集団分布の標準偏差リスト
        """
        self.X = X
        self.y = y
        self.X_mu = np.array(X_mu)
        self.X_sigma = np.array(X_sigma)
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

    def sample_mean(self) -> None:
        value = self.y.mean()
        return value
    
    def maximum_likelihood(self, draw) -> None:
        self.draw = draw
        # パラメータの学習
        model = LogisticRegression(solver='lbfgs', max_iter=1000, penalty = None)
        model.fit(self.X, self.y)

        # パラメータの取得
        coefs = model.coef_[0]
        
        # pの推定
        ## Xの生成
        intercept_column = np.ones((self.draw, 1))
        X_list = []
        for i in range(self.n_features-1):
            X_i = np.random.normal(loc=self.X_mu[i], scale=self.X_sigma[i], size=self.draw)
            X_list.append(X_i)
        X_list.insert(0, intercept_column.flatten())
        X_true = np.array(X_list)
        ## Xとパラメータを乗算
        beta_x = coefs @ X_true
        P_y_list = 1 /(1 + np.exp(-beta_x))
        P_y = P_y_list.mean()

        return P_y