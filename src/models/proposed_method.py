import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
from polyagamma import random_polyagamma

class proposed_method:
    def __init__(self, X: np.array, y: np.array, b_0: list, B_0: list, n_classes: int, X_mu: list, X_sigma: list, burn: int, draw: int) -> None:
        """
        :param X: サンプリング後のX
        :param y: サンプリング後のy
        :param b_0: 各クラスの事前分布の平均ベクトル
        :param b_0: 各クラスの事前分布の分散共分散行列
        :param X_mu: 共変量の母集団分布の平均リスト
        :param X_sigma: 共変量の母集団分布の標準偏差リスト
        """
        self.X = X
        self.y = y
        self.B_0 = np.array(B_0)
        self.b_0 = np.array(b_0)
        self.X_mu = np.array(X_mu)
        self.X_sigma = np.array(X_sigma)
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.n_classes = n_classes
        self.burn = burn
        self.draw = draw

    def _safe_log(self, x, epsilon=1e-10):
        """Log with safety to avoid log(0) or log of negative numbers."""
        x_safe = np.maximum(x, epsilon)
        return np.log(x_safe)

    def _safe_exp(self, x, max_val=1e2):
        """Exp with clipping to avoid overflow."""
        x_clipped = np.clip(x, -max_val, max_val)
        return np.exp(x_clipped)

    def _safe_matrix(self, matrix):
        """Ensure no NaN or inf in the matrix."""
        return np.nan_to_num(matrix, nan=0.0, posinf=1e10, neginf=-1e10)

    def sample_beta(self):
        """サンプリング"""
        beta_strage = []

        # 初期化
        par_beta = np.zeros((self.n_classes, self.n_features))
        par_C = np.full((self.n_classes, self.n_samples), np.log(self.n_classes - 1))
        par_eta = par_beta @ self.X.T - par_C

        num = 1
        B_inv = [np.linalg.inv(self.B_0[j]) for j in range(self.n_classes)]

        while num <= (self.burn + self.draw):
            beta_strage_class = []
            par_w = []

            # w_ijのサンプリング
            for j in range(self.n_classes):
                par_w_tmp = random_polyagamma(1, z=par_eta[j], size=(1, self.n_samples))[0]
                par_w.append(par_w_tmp)

            for j in range(self.n_classes):
                par_w_j = par_w[j].reshape(-1, 1)
                par_kappa_j = (self.y - 0.5).reshape(-1, 1)
                par_z_j = (par_kappa_j / par_w_j).reshape(-1, 1)
                par_W_j = self._safe_matrix(np.diag(par_w_j.flatten()))
                B = np.linalg.inv(self.X.T @ par_W_j @ self.X + B_inv[j])

                par_C_j = par_C[j].reshape(-1, 1)
                b_0_j = self.b_0[j].reshape(-1, 1)
                b = B @ (self.X.T @ par_W_j @ (par_C_j + par_z_j) + B_inv[j] @ b_0_j)

                # パラメータの更新
                b = b.flatten()
                par_beta[j] = np.random.multivariate_normal(b, B, size=1)[0]
                beta_strage_class.append(par_beta[j])

            for j in range(self.n_classes):
                class_list = list(range(self.n_classes))
                class_list.remove(j)
                X_beta_tmp = 0
                for j_C in class_list:
                    par_beta_tmp = par_beta[j_C].reshape(-1, 1)
                    X_beta_tmp += self._safe_exp(self.X @ par_beta_tmp)
                par_C[j] = self._safe_log(X_beta_tmp.flatten())
                par_beta_tmp = par_beta[j].reshape(-1, 1)
                par_eta[j] = (self.X @ par_beta_tmp - par_C[j].reshape(-1, 1)).flatten()

            beta_strage.append(np.array(beta_strage_class))
            num += 1

        beta_posterior = beta_strage[self.burn:]
        self.beta_posterior = beta_posterior

        return beta_posterior

        
    
    def estimate(self):
        # 切片項を追加するために、全て1の列ベクトルを作成
        intercept_column = np.ones((self.draw, 1))
        # X を正規分布から生成
        X_list = []
        for i in range(self.n_features-1):
            X_i = np.random.normal(loc=self.X_mu[i], scale=self.X_sigma[i], size=self.draw)
            X_list.append(X_i)
        # 全て1の列ベクトルを先頭に挿入
        X_list.insert(0, intercept_column.flatten())
        X_true = np.array(X_list).T
        beta_x = np.sum(self.beta_posterior*X_true, axis=1)
        P_y_list = 1 /(1 + np.exp(-beta_x))
        P_y = P_y_list.mean()

        return P_y

        
class proposed_method_binomial:
    def __init__(self, X: np.array, y: np.array, b_0: list, B_0: list, X_mu: list, X_sigma: list, burn: int, draw: int) -> None:
        """
        :param X: サンプリング後のX
        :param y: サンプリング後のy
        :param b_0: 各クラスの事前分布の平均ベクトル
        :param b_0: 各クラスの事前分布の分散共分散行列
        :param X_mu: 共変量の母集団分布の平均リスト
        :param X_sigma: 共変量の母集団分布の標準偏差リスト
        """
        self.X = X
        self.y = y
        self.B_0 = np.array(B_0)
        self.b_0 = np.array(b_0)
        self.X_mu = np.array(X_mu)
        self.X_sigma = np.array(X_sigma)
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.burn = burn
        self.draw = draw
    
    def _safe_log(self, x, epsilon=1e-10):
        """Log with safety to avoid log(0) or log of negative numbers."""
        x_safe = np.maximum(x, epsilon)
        return np.log(x_safe)

    def _safe_exp(self, x, max_val=1e2):
        """Exp with clipping to avoid overflow."""
        x_clipped = np.clip(x, -max_val, max_val)
        return np.exp(x_clipped)

    def _safe_matrix(self, matrix):
        """Ensure no NaN or inf in the matrix."""
        return np.nan_to_num(matrix, nan=0.0, posinf=1e10, neginf=-1e10)

    def sample_beta(self):
        """サンプリング"""
        beta_strage = np.zeros(self.n_features)

        # 初期化
        par_beta = np.zeros(self.n_features)
        par_eta = self.X @ par_beta
        #print(f"par_beta shape: {par_beta.shape}, par_eta shape: {par_eta.shape}") # デバッグ用出力

        num = 1
        B_inv = np.linalg.inv(self.B_0)[0]
        #print(f"B_inv shape: {B_inv.shape}") # デバッグ用出力

        while num <= (self.burn + self.draw):

            # w_iのサンプリング
            par_w = random_polyagamma(1, z=par_eta, size=(1, self.n_samples)).flatten()
            par_W = self._safe_matrix(np.diag(par_w))
            #print(f"par_w shape: {par_w.shape}, par_W shape: {par_W.shape}") # デバッグ用出力
            par_kappa = (self.y - 0.5)
            par_z = (par_kappa / par_w).flatten()
            #print(f"par_kappa shape: {par_kappa.shape}, par_z shape: {par_z.shape}") # デバッグ用出力
            
            B = np.linalg.inv((self.X.T @ par_W @ self.X) + B_inv)
            #print(f"B shape: {B.shape}") # デバッグ用出力

            b = B @ (self.X.T @ par_W @ par_z + B_inv @ self.b_0)
            #print(f"b shape: {b.shape}") # デバッグ用出力

            # パラメータの更新
            par_beta = np.random.multivariate_normal(b, B, size=1)[0]
            par_eta = self.X @ par_beta
            #print(f"Iteration {num}: par_beta shape: {par_beta.shape}, par_eta shape: {par_eta.shape}") # デバッグ用出力

            beta_strage = np.vstack((beta_strage, par_beta))
            num += 1

        beta_posterior = beta_strage[self.burn:]
        self.beta_posterior = beta_posterior
        #print(f"beta_posterior shape: {beta_posterior.shape}") # デバッグ用出力
        

        return beta_posterior

