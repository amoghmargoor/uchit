from scipy.stats import norm

import itertools
import math
import numpy as np
import sys


class GaussianModel:

    # ToDO - Fix the values initialisation
    alpha = 2.0
    beta = np.array([1, 1, 1, 1, 1, 1, 1]) * pow(10, -6)
    gamma = np.array([1, 0.01, 0.5, 1, 1, 1, 0.01])
    theta = np.array([1.0, 0.09, 0.17, 0.17, 0.17, 0.17, 0.8])
    training_pair_wise_corr = None

    def __init__(self, training_data, training_out):
        self.training_data = training_data
        self.training_out = training_out
        self.get_corr_with_train_data()
        self.best_out = math.min(training_out)

    def train(self):
        # ToDo: Implement a train function to find precise values of alpha, beta and gamma
        pass

    def add_sample_to_train_data(self, config, out):
        self.training_pair_wise_corr = None
        self.training_data.append(config)
        self.training_out.append(out)

    def get_best_config(self, normalized_configs):
        best_config = None
        best_out = sys.maxint
        for config in list(itertools.product(*normalized_configs)):
            out = self.predict(config)
            if out < best_out:
                best_out = out
                best_config = config
        return best_config

    def get_correlation(self, var1, var2):
        correlation = 1
        for i in range(0, len(var1)):
            term = math.exp(-self.theta[i] * pow(abs(var1[i] - var2[i]), self.gamma[i]))
            correlation = correlation * term
        return correlation

    def get_training_pairwise_correlation(self):
        if self.training_pair_wise_corr is None:
            metrics = []
            for i in range(0, len(self.training_data)):
                metrics.append([])
                for j in range(0, len(self.training_data)):
                    metrics[i].append(
                        self.get_correlation(self.training_data[i], self.training_data[j]))
            self.training_pair_wise_corr = np.array(metrics)

        return self.training_pair_wise_corr

    def get_correlation_with_train_data(self, config):
        metrics = []
        for i in range(0, len(self.training_data)):
            metrics.append([])
            metrics[i].append(self.get_correlation(config, self.training_data[i]))
        return np.array(metrics)

    def get_training_params(self):
        return self.training_data

    def get_mean(self, config):
        term1 = np.dot(config, self.beta)
        term2 = self.training_out - np.dot(self.get_training_params(), self.beta)
        term3 = np.dot(self.get_correlation_with_train_data(config).transpose(),
                       np.linalg.inv(self.get_training_pairwise_correlation()))
        term4 = np.dot(term3, term2)
        return term1 + term4

    def get_variance(self, config):
        corr_with_train_data = self.get_correlation_with_train_data(config)
        corr_pairwise_train_data = self.get_training_pairwise_correlation()
        term1 = np.dot(corr_with_train_data.transpose(), np.linalg.inv(corr_pairwise_train_data))
        term2 = np.dot(term1, corr_with_train_data)
        term3 = 1 - term2
        return np.linalg.det(pow(self.alpha, 2) * term3)

    def get_mu(self, config):
        return (self.best_out - self.get_mean(config)) / math.sqrt(self.get_variance(config))

    def predict(self, config):
        mu = self.get_mu(config)
        return math.sqrt(self.get_variance(config)) * (mu * norm.cdf(mu) + norm.pdf(mu))
