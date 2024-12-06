# 任意の次元のyとXを生成
# Xを生成してからyを生成
# Xの値を元にしてデータをバイアスサンプリング

import numpy as np
import pandas as pd

class GenerateData:
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        X_mu_list: list,
        X_sigma_list: list,
        beta_mu_matrix: list,
        beta_sigma_matrix: list
    ) -> None:
        """
        :param n_features: 特徴量の次元数（切片を除く）
        :param n_classes: クラス数（2以上の整数）
        :param X_mu_list: 特徴量の母集団分布の平均リスト
        :param X_sigma_list: 特徴量の母集団分布の標準偏差リスト
        :param beta_mu_matrix: パラメータの母集団分布の平均のリスト（各クラスごとのリスト）
        :param beta_sigma_matrix: パラメータの母集団分布の標準偏差のリスト（各クラスごとのリスト）
        """
        self.n_population = 100  # サンプル生成のため適当に大きい数で母集団を生成
        self.n_features = n_features
        self.n_classes = n_classes
        self.X_mu_list = X_mu_list
        self.X_sigma_list = X_sigma_list
        self.beta_mu_matrix = beta_mu_matrix
        self.beta_sigma_matrix = beta_sigma_matrix
        self.X = None
        self.y = None

    def generate_non_bias_data(self):
        """
        正規分布から X を生成し、多項ロジスティック回帰モデルに基づいて y を生成する。
        """
        # 切片項を追加するために、全て1の列ベクトルを作成
        intercept_column = np.ones((self.n_population, 1))

        # X を正規分布から生成
        X_list = []
        for i in range(self.n_features):
            X_i = np.random.normal(
                loc=self.X_mu_list[i],
                scale=self.X_sigma_list[i],
                size=self.n_population
            )
            X_list.append(X_i)
        # 全て1の列ベクトルを先頭に挿入
        X_list.insert(0, intercept_column.flatten())
        self.X = np.array(X_list).T  # 形状: (n_population, n_features + 1)

        # ロジスティック回帰モデルの真のパラメータを分布から生成
        beta = []
        for k in range(self.n_classes):
            beta_k = []
            for i in range(self.n_features + 1):
                beta_mu = self.beta_mu_matrix[k][i]
                beta_sigma = self.beta_sigma_matrix[k][i]
                beta_ki = np.random.normal(
                    loc=beta_mu,
                    scale=beta_sigma,
                    size=1
                )
                beta_k.append(beta_ki[0])  # size=1なので、値を取り出す
            beta.append(beta_k)
        beta = np.array(beta)  # 形状: (n_classes, n_features + 1)

        # 線形予測子を計算
        eta = np.dot(self.X, beta.T)  # 形状: (n_population, n_classes)

        # ソフトマックス関数で確率に変換
        def softmax(z):
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 数値安定化
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)

        probabilities = softmax(eta)  # 形状: (n_population, n_classes)

        # y を生成（確率に基づいてクラスをサンプリング）
        self.y = np.array([
            np.random.choice(self.n_classes, p=probabilities[i])
            for i in range(self.n_population)
        ])

        return self.X, self.y

    def generate_bias_data_deterministic(self, n_samples: int, X_features_bias: list, centers_of_sampling: list, window_sizes: list):
        """
        X の値に基づいて決定的にサンプリング
        指定された特徴量 `X_features_bias` の値が、それぞれ `centers_of_sampling` を中心とした `window_sizes` の範囲内にあるデータのみを抽出します。

        :param n_samples: サンプルサイズ（バイアスサンプリング後）
        :param X_features_bias: バイアスをかける特徴量のインデックスのリスト
        :param centers_of_sampling: サンプリングの中心となる値のリスト
        :param window_sizes: サンプリングウィンドウのサイズのリスト
        :return: バイアスサンプリングされた X と y
        """
        self.n_samples = n_samples
        if self.X is None or self.y is None:
            raise ValueError("データを生成してからバイアスサンプリングを行ってください。")
        
        df = pd.DataFrame(np.concatenate((self.X, self.y.reshape(-1, 1)), axis=1))
        col_names = [f"X_{num}" for num in range(self.n_features + 1)]
        col_names.append("y")
        df.columns = col_names

        if not (len(X_features_bias) == len(centers_of_sampling) == len(window_sizes)):
            raise ValueError("X_features_bias、centers_of_sampling、window_sizes の長さは同じでなければなりません。")

        # フィルタリング条件を作成
        condition = pd.Series([True] * len(df))
        for feature_index, center, window in zip(X_features_bias, centers_of_sampling, window_sizes):
            upper_value = center + window / 2
            lower_value = center - window / 2
            condition &= (df[f"X_{feature_index}"] >= lower_value) & (df[f"X_{feature_index}"] <= upper_value)
        
        df_filtered = df[condition].reset_index(drop=True)
        df_filtered = df_filtered.loc[:self.n_samples - 1]
        biased_X = df_filtered.iloc[:, :-1].values
        biased_y = df_filtered.iloc[:, -1].values.astype(int)

        return biased_X, biased_y

    
    def generate_bias_data_probablistic(self, X_feature_bias: int, center_of_sampling: int, center_bias: int):
        """
        確率的にバイアスサンプリング
        指定された特徴量 `X_feature_bias` の値が `center_of_sampling` に近いほど、高い確率でサンプリングされます。

        :param X_feature_bias: バイアスをかける特徴量のインデックス
        :param center_of_sampling: サンプリングの中心となる値
        :param center_bias: 中心からの距離に基づいて確率を調整するパラメータ
        :return: バイアスサンプリングされた X と y
        """
        if self.X is None or self.y is None:
            raise ValueError("データを生成してからバイアスサンプリングを行ってください。")

        df = pd.DataFrame(np.concatenate((self.X, self.y.reshape(-1, 1)), axis=1))
        col_names = [f"X_{num}" for num in range(self.n_features + 1)]
        col_names.append(["y"])
        distances = np.abs(df[f"X_{X_feature_bias}"] - center_of_sampling)
        probabilities = np.exp(-center_bias * distances)
        probabilities /= np.sum(probabilities)  # 確率を正規化

        sampled_indices = np.random.choice(len(df), size=len(df), replace=True, p=probabilities)
        biased_X = df.iloc[sampled_indices, :-1].values
        biased_y = df.iloc[sampled_indices, -1].values

        return biased_X, biased_y


class GenerateData_binomial:
    def __init__(
        self,
        n_features: int,
        # n_classes: int,  # クラス数は2なので削除
        X_mu_list: list,
        X_sigma_list: list,
        beta_mu: list,  # beta_mu_matrix をベクトルに変更
        beta_sigma: list,  # beta_sigma_matrix をベクトルに変更
    ) -> None:
        """
        :param n_features: 特徴量の次元数（切片を除く）
        :param X_mu_list: 特徴量の母集団分布の平均リスト
        :param X_sigma_list: 特徴量の母集団分布の標準偏差リスト
        :param beta_mu: パラメータの母集団分布の平均のリスト
        :param beta_sigma: パラメータの母集団分布の標準偏差のリスト
        """
        self.n_population = 1000000  # サンプル生成のため適当に大きい数で母集団を生成
        self.n_features = n_features
        # self.n_classes = n_classes  # クラス数は2なので削除
        self.X_mu_list = X_mu_list
        self.X_sigma_list = X_sigma_list
        self.beta_mu = beta_mu  # beta_mu_matrix をベクトルに変更
        self.beta_sigma = beta_sigma  # beta_sigma_matrix をベクトルに変更
        self.X = None
        self.y = None

    def generate_non_bias_data(self):
        """
        正規分布から X を生成し、2項ロジスティック回帰モデルに基づいて y を生成する。
        """
        # 切片項を追加するために、全て1の列ベクトルを作成
        intercept_column = np.ones((self.n_population, 1))

        # X を正規分布から生成
        X_list = []
        for i in range(self.n_features):
            X_i = np.random.normal(
                loc=self.X_mu_list[i],
                scale=self.X_sigma_list[i],
                size=self.n_population
            )
            X_list.append(X_i)
        # 全て1の列ベクトルを先頭に挿入
        X_list.insert(0, intercept_column.flatten())
        self.X = np.array(X_list).T  # 形状: (n_population, n_features + 1)

        # 2項ロジスティック回帰モデルの真のパラメータを分布から生成
        # 個体ごとに真のパラメータが異なるようにする（ベイズ的な設定）
        beta = []
        for i in range(self.n_features + 1):
            beta_mu = self.beta_mu[i]
            beta_sigma = self.beta_sigma[i]
            beta_i = np.random.normal(
                loc=beta_mu,
                scale=beta_sigma,
                size=self.n_population
            )
            beta.append(beta_i)  # size=1なので、値を取り出す
        beta = np.array(beta).T  # 形状: (n_features + 1,)

        # 線形予測子を計算
        eta = np.sum(self.X * beta, axis=1)  # 形状: (n_population,)

        # シグモイド関数で確率に変換
        probabilities = 1 / (1 + np.exp(-eta))  # 形状: (n_population,)

        # y を生成（確率に基づいてクラスをサンプリング）
        self.y = np.random.binomial(1, probabilities) # 2項分布からサンプリング

        """
        # 2項ロジスティック回帰モデルの真のパラメータを分布から生成
        # 個体ごとに真のパラメータが異なるようにする（ベイズ的な設定）
        beta = []
        for i in range(self.n_features + 1):
            beta_mu = self.beta_mu[i]
            beta_sigma = self.beta_sigma[i]
            beta_i = np.random.normal(
                loc=beta_mu,
                scale=beta_sigma,
                size=1
            )
            beta.append(beta_i[0])  # size=1なので、値を取り出す
        beta = np.array(beta)  # 形状: (n_features + 1,)

        # 線形予測子を計算
        eta = self.X * beta  # 形状: (n_population,)
        # シグモイド関数で確率に変換
        probabilities = 1 / (1 + np.exp(-eta))  # 形状: (n_population,)

        # y を生成（確率に基づいてクラスをサンプリング）
        self.y = np.random.binomial(1, probabilities) # 2項分布からサンプリング
        """

        return self.X, self.y

    def generate_bias_data_deterministic(
            self, 
            n_samples: int, 
            X_features_bias: int, 
            centers_of_sampling: int,
            window_sizes: int,
            sample_type:str = "upper"
            ):
        """
        X の値に基づいて決定的にサンプリング
        指定された特徴量 `X_features_bias` の値が、それぞれ `centers_of_sampling` を中心とした `window_sizes` の範囲内にあるデータのみを抽出します。

        :param n_samples: サンプルサイズ（バイアスサンプリング後）
        :param X_features_bias: バイアスをかける特徴量のインデックスのリスト
        :param centers_of_sampling: サンプリングの中心となる値のリスト
        :param window_sizes: サンプリングウィンドウのサイズのリスト
        :return: バイアスサンプリングされた X と y
        """
        self.n_samples = n_samples
        if self.X is None or self.y is None:
            raise ValueError("データを生成してからバイアスサンプリングを行ってください。")
        
        df = pd.DataFrame(np.concatenate((self.X, self.y.reshape(-1, 1)), axis=1))
        col_names = [f"X_{num}" for num in range(self.n_features + 1)]
        col_names.append("y")
        df.columns = col_names

        bias_col = f"X_{X_features_bias}"

        if sample_type == "window":

            df_bias = df.loc[
                            (df[bias_col] <= centers_of_sampling + window_sizes/2) &
                            (df[bias_col] >= centers_of_sampling - window_sizes/2)
                            ]
        elif sample_type == "lower":
            df_bias = df.loc[
                            (df[bias_col] <= centers_of_sampling)
                            ]
        elif sample_type == "upper":
            df_bias = df.loc[
                            (df[bias_col] >= centers_of_sampling)
                            ]
            
        biased_X = df_bias[df_bias.columns[:-1]].sample(n_samples)
        biased_y = df_bias["y"].sample(n_samples)

        df_bias

        return np.array(biased_X), np.array(biased_y)

    
    def generate_bias_data_probablistic(self, X_feature_bias: int, center_of_sampling: int, center_bias: int):
        """
        確率的にバイアスサンプリング
        指定された特徴量 `X_feature_bias` の値が `center_of_sampling` に近いほど、高い確率でサンプリングされます。

        :param X_feature_bias: バイアスをかける特徴量のインデックス
        :param center_of_sampling: サンプリングの中心となる値
        :param center_bias: 中心からの距離に基づいて確率を調整するパラメータ
        :return: バイアスサンプリングされた X と y
        """
        if self.X is None or self.y is None:
            raise ValueError("データを生成してからバイアスサンプリングを行ってください。")

        df = pd.DataFrame(np.concatenate((self.X, self.y.reshape(-1, 1)), axis=1))
        col_names = [f"X_{num}" for num in range(self.n_features + 1)]
        col_names.append(["y"])
        distances = np.abs(df[f"X_{X_feature_bias}"] - center_of_sampling)
        probabilities = np.exp(-center_bias * distances)
        probabilities /= np.sum(probabilities)  # 確率を正規化

        sampled_indices = np.random.choice(len(df), size=len(df), replace=True, p=probabilities)
        biased_X = df.iloc[sampled_indices, :-1].values
        biased_y = df.iloc[sampled_indices, -1].values

        return biased_X, biased_y
    
    def generate_bias_data(
            self, 
            n_samples: int, 
            X_features_bias: int, 
            threshfold: int = None,  # threshfold をオプション引数にする
            sample_type:str = "linear",
            ):
        self.n_samples = n_samples
        if self.X is None or self.y is None:
            raise ValueError("データを生成してからバイアスサンプリングを行ってください。")

        if "threshfold" in sample_type and threshfold is None:
            raise ValueError("sample_type が 'threshfold' を含む場合、threshfold は必須です。")
        
        # データフレームの作成
        df = pd.DataFrame(np.concatenate((self.X, self.y.reshape(-1, 1)), axis=1))
        col_names = [f"X_{num}" for num in range(self.n_features + 1)]
        col_names.append("y")
        df.columns = col_names
        bias_col = f"X_{X_features_bias}"

        # 線形の場合
        if sample_type == "linear":
            sum_ = df[bias_col].sum()
            min_ = df[bias_col].min()
            df["weight"] = df[bias_col].apply(lambda x: (x-min_)/sum_)
            sampled_df = df.sample(n=self.n_samples, weights="weight", replace=True)
            sampled_X = sampled_df.drop(columns=["y", "weight"]).values
            sampled_y = sampled_df["y"].values
            
            return sampled_X, sampled_y


        # 非線形の場合
        elif sample_type == "non_linear":
            beta = 1 # ロジスティク回帰のパラメータ．大きいほど値の大きいものをサンプリングしやすい
            weights = 1 / (1 + np.exp(-(beta * df[bias_col].values)))
            probs = weights / weights.sum()
            sampled_df = df.sample(n=self.n_samples, weights=probs, replace=True)
            sampled_X = sampled_df.drop(columns=["y"]).values
            sampled_y = sampled_df["y"].values
            return sampled_X, sampled_y


        # 閾値使用かつステップ的な場合
        elif sample_type == "threshfold_step":
            prob = 0.7  # 確率probで閾値以上をサンプリングし、1-probで以下をサンプリングする
            targets = np.random.binomial(1, prob, size=self.n_samples)
            upper_sample_num = np.count_nonzero(targets)
            lower_sample_num = self.n_samples - upper_sample_num
            df_upper = df.loc[df[bias_col] >= threshfold].sample(upper_sample_num)
            df_lower = df.loc[df[bias_col] < threshfold].sample(lower_sample_num)
            sampled_df = pd.concat([df_lower,df_upper])
            sampled_X = sampled_df.drop(columns=["y"]).values
            sampled_y = sampled_df["y"].values

            return sampled_X, sampled_y
            
        # 閾値使用かつ非線形な場合
        elif sample_type == "threshfold_smooth":
            beta_1 = 0.5
            beta_0 = -threshfold * beta_1
            weights = 1 / (1 + np.exp(-(beta_0+beta_1*df[bias_col].values)))
            probs = weights / weights.sum()
            sampled_df = df.sample(n=self.n_samples, weights=probs, replace=True)
            sampled_X = sampled_df.drop(columns=["y"]).values
            sampled_y = sampled_df["y"].values
            return sampled_X, sampled_y
            

        

        

        

