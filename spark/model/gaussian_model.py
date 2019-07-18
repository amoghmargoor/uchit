from scipy.stats import norm

import itertools
import math
import numpy as np
import sys

from spark.discretizer.lhs_discrete_sampler import LhsDiscreteSampler
from spark.discretizer.normalizer import ConfigNormalizer, ConfigDenormalizer


class GaussianModel:

    # ToDO - Fix the values initialisation
    alpha = 2.0
    beta = np.array([1, 1, 1, 1, 1, 1, 1]) * pow(10, -6)
    gamma = np.array([1, 0.01, 0.5, 1, 1, 1, 0.01])
    theta = np.array([1.0, 0.09, 0.17, 0.17, 0.17, 0.17, 0.8])

    def __init__(self, configs):
        self.training_pair_wise_corr = None
        self.training_inp = []
        self.training_out = []
        self.training_inp_normalized = []
        self.training_conf_names = []
        self.best_out = None
        self.configs = configs

    def train(self):
        if not self.configs.configs_elements.config_elements_list:
            raise Exception("No training data found")

        for ele in self.configs.configs_elements.config_elements_list:
            self.training_conf_names = list(map(lambda x: x[0], ele["configs"]))
            self.training_inp += list(map(lambda x: x[1], ele["configs"]))
            self.training_inp_normalized += list(map(lambda x: x[1].get_normalized_value(), ele["configs"]))
            self.training_out += ele["out"]
        self.best_out = math.min(self.training_out)
        # ToDo: Implement a train function to find precise values of alpha, beta and gamma

    def add_sample_to_train_data(self, config, out):
        self.training_pair_wise_corr = None
        self.training_data.append(config)
        self.training_out.append(out)

    def method_to_return_normalized_values(self):
        # Normalize the values
        # Use LHS to get the cprrect values
        normalizer = ConfigNormalizer(self.configs)
        lhs_sampler = LhsDiscreteSampler(normalizer._normalized_config, 2)
        return lhs_sampler._get_samples(2)

    def get_best_config(self):
        if self.training_inp_normalized is None:
            raise Exception("No training data found")

        normalized_values = self.method_to_return_normalized_values()
        best_config_value = None
        best_config = {}
        best_out = sys.maxint
        for config in list(itertools.product(*normalized_values)):
            out = self.predict(config)
            if out < best_out:
                best_out = out
                best_config_value = config
        for name in self.training_conf_names():
            config_value = self.configs[name]
            best_config[name] = ConfigDenormalizer.denormalize(
                best_config_value,
                config_value.get_min_for_normalization(),
                config_value.get_max_for_normalization()
            )
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
            for i in range(0, len(self.training_inp_normalized)):
                metrics.append([])
                for j in range(0, len(self.training_inp_normalized)):
                    metrics[i].append(
                        self.get_correlation(self.training_inp_normalized[i], self.training_inp_normalized[j]))
            self.training_inp_normalized = np.array(metrics)

        return self.training_pair_wise_corr

    def get_correlation_with_train_data(self, config):
        metrics = []
        for i in range(0, len(self.training_inp_normalized)):
            metrics.append([])
            metrics[i].append(self.get_correlation(config, self.training_inp_normalized[i]))
        return np.array(metrics)

    def get_training_params(self):
        return self.training_inp_normalized

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
