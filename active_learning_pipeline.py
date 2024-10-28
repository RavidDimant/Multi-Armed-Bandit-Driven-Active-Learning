import copy
import math
import time

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import entropy
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import random
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def generate_plot(accuracy_scores_dict):
    """
    Generate a plot with specified figure size
    """
    plt.figure(figsize=(15, 9))  # Adjust width and height as desired
    for criterion, accuracy_scores in accuracy_scores_dict.items():
        plt.plot(range(1, len(accuracy_scores) + 1), accuracy_scores, label=criterion)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    # plt.title('')
    plt.legend()
    plt.show()


class UCB:
    def __init__(self, n_arms, c=1):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.mus = np.zeros(n_arms)
        self.tot_rounds = 0

    def choose_arm(self):
        ucb = lambda mu_i, n_i, tot: mu_i + self.c * math.sqrt((math.log(self.tot_rounds) / n_i))
        ucb_scores = [ucb(self.mus[i], self.counts[i], self.tot_rounds) for i in range(self.n_arms)]
        return np.argmax(ucb_scores)

    def update(self, arm, reward):
        self.tot_rounds += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        self.mus[arm] = ((n - 1) / n) * self.mus[arm] + (1 / n) * reward



class ActiveLearningPipeline:

    def __init__(self, iterations, budget_per_iter, data_path, train_label_test_split: tuple):

        self.model = None  # model

        self.iterations = iterations
        self.budget_per_iter = budget_per_iter

        # read and prepare train data, data to label and test data
        data_df = pd.read_csv(data_path)
        data_df = shuffle(data_df, random_state=42)

        labeled_df = data_df.iloc[:train_label_test_split[0]]
        unlabeled_df = data_df.iloc[train_label_test_split[0]:train_label_test_split[1]]
        test_df = data_df.iloc[train_label_test_split[1]:train_label_test_split[2]]

        labeled_data = {'x': [np.array(row) for row in labeled_df.drop('Diabetes', axis=1).to_numpy()],
                      'y': np.array([l for l in labeled_df['Diabetes']])}

        unlabeled_data = {'x': [np.array(row) for row in unlabeled_df.drop('Diabetes', axis=1).to_numpy()],
                      'y':  np.array([l for l in unlabeled_df['Diabetes']])}

        test_data = {'x': [np.array(row) for row in test_df.drop('Diabetes', axis=1).to_numpy()],
                     'y':  np.array([l for l in test_df['Diabetes']])}

        self.data = {'labeled_data': labeled_data, 'unlabeled_data': unlabeled_data, 'test_data': test_data}
        self.copy_data = {}

        self.features = np.array(data_df.drop('Diabetes', axis=1).columns)

    " Auxiliary methods "

    def get_data_copy(self):
        labeled_data_x = copy.deepcopy(self.data['labeled_data']['x'])
        labeled_data_y = copy.deepcopy(self.data['labeled_data']['y'])

        unlabeled_data_x = copy.deepcopy(self.data['unlabeled_data']['x'])
        unlabeled_data_y = copy.deepcopy(self.data['unlabeled_data']['y'])

        test_data_x = copy.deepcopy(self.data['test_data']['x'])
        test_data_y = copy.deepcopy(self.data['test_data']['y'])

        return labeled_data_x, labeled_data_y, unlabeled_data_x, unlabeled_data_y, test_data_x, test_data_y

    " Sampling Methods "

    # TODO:
    #   - maybe send n_select instead of computing it (for MAB could use n_select = 1)

    def _random_sampling(self, predicted_probabilities):
        pool_size = len(predicted_probabilities)
        to_label_size = pool_size
        n_select = min(self.budget_per_iter, to_label_size)
        # Randomly select n_select indices without replacement
        selected_indices = random.sample(range(to_label_size), n_select)

        return selected_indices

    def _uncertainty_sampling(self, predicted_probabilities):
        to_label_size = len(predicted_probabilities)
        n_select = min(self.budget_per_iter, to_label_size)
        uncertainties = entropy(predicted_probabilities.T)
        selected_indices = np.argsort(uncertainties)[-n_select:].astype(int).tolist()
        selected_indices = [i for i in reversed(selected_indices)]

        return selected_indices

    # ________________

    # def _worst_similarity_sampling(self, _):
    #     labeled_x, labeled_y, unlabeled_x, unlabeled_y, _, _ = self.get_data_copy()
    #
    #     loss_func = lambda pred_probs, real: -math.log(pred_probs[real] + 1e-3)
    #
    #     # use model to predict prob of train
    #     y_pred = self.model.predict_proba(labeled_x)
    #
    #     loss = [loss_func(p, y) for p, y in zip(y_pred, labeled_y)]
    #     worst_indices = np.argsort(loss)[::-1]
    #
    #     selected_indices = []
    #     for ind in worst_indices:
    #
    #         similarities = cosine_similarity(labeled_x[ind].reshape(1, -1), np.array(unlabeled_x))[0]
    #         most_similar = np.argsort(similarities)[::-1]
    #         print(most_similar)
    #         exit(0)
    #         for s in most_similar:
    #             if s not in selected_indices:
    #                 selected_indices.append(s)
    #                 break
    #
    #         if len(selected_indices) == self.budget_per_iter:
    #             break
    #
    #     #
    #     print(len(selected_indices))
    #     # print(selected_indices)
    #     # print("ok")
    #     # exit(0)
    #
    #     return selected_indices

    def _diversity_sampling(self, _):
        # Calculate pairwise distances between samples
        from sklearn.metrics import pairwise_distances
        distance_matrix = pairwise_distances(self.copy_data['unlabeled_data']['x'])

        # Greedily select samples to maximize diversity
        selected_indices = []
        while len(selected_indices) < self.budget_per_iter:
            if not selected_indices:
                # Start with a random sample
                selected_indices.append(np.random.choice(range(len(self.copy_data['unlabeled_data']['x']))))
            else:
                # Calculate the minimum distance of each sample to already selected ones
                min_distances = np.min(distance_matrix[:, selected_indices], axis=1)
                # Select the sample with the maximum minimum distance
                next_index = np.argmax(min_distances)
                selected_indices.append(next_index)
        return selected_indices

    def _density_weighted_uncertainty_sampling(self, predicted_probabilities):
        uncertainties = entropy(predicted_probabilities.T)
        # Calculate sample density in the feature space
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(self.copy_data['unlabeled_data']['x'])
        densities = np.sum(np.exp(-distances), axis=1)  # Higher density for closer points

        # Combine uncertainty and density
        combined_scores = uncertainties * densities
        selected_indices = list(np.argsort(combined_scores)[-self.budget_per_iter:])
        return selected_indices

    def _margin_sampling(self, predicted_probabilities):
        # Calculate the margin between the top two probabilities
        sorted_probs = np.sort(predicted_probabilities, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]

        # Select samples with the smallest margins (closest to the decision boundary)
        selected_indices = list(np.argsort(margins)[:self.budget_per_iter])
        # if not isinstance(selected_indices, list):
        #     print("_margin_sampling")
        #     exit(0)
        return selected_indices

    def risk_based_sampling(self, y_pred):
        # Calculate uncertainty (entropy)
        uncertainties = entropy(y_pred.T)

        # Extract important feature columns (BMI, Smoking, Heart Disease...)
        features_list = list(self.features)
        bmi_index = features_list.index('BMI')
        elderly_age1_index = features_list.index('Age_Category_80+')
        elderly_age2_index = features_list.index('Age_Category_75-79')
        elderly_age3_index = features_list.index('Age_Category_70-74')
        heart_disease_index = features_list.index('Heart_Disease')
        smoking_history_index = features_list.index('Smoking_History')
        health_condition1_index = features_list.index('General_Health_Excellent')
        health_condition2_index = features_list.index('General_Health_Fair')
        health_condition3_index = features_list.index('General_Health_Good')
        health_condition4_index = features_list.index('General_Health_Poor')
        health_condition5_index = features_list.index('General_Health_Very Good')
        arthritis_index = features_list.index('Arthritis')
        exercise_index = features_list.index('Exercise')
        sex_index = features_list.index('Sex')
        skin_cancer_index = features_list.index('Skin_Cancer')

        # Compute risk score: higher BMI, older age, presence of heart disease or smoking history increase risk
        unlabeled_x = np.array(self.copy_data['unlabeled_data']['x'])
        bmi_normalized = (unlabeled_x[:, bmi_index] - 15) / (40 - 15)  # Rescale BMI to 0-1 range
        risk_scores = (bmi_normalized * 0.3 +  # Weight BMI more
                       (unlabeled_x[:, elderly_age1_index] +  # Age contributes as well
                        unlabeled_x[:, elderly_age2_index] +
                        unlabeled_x[:, elderly_age3_index]) * 0.1 +
                       unlabeled_x[:, heart_disease_index] * 0.3 +  # Heart Disease
                       unlabeled_x[:, smoking_history_index] * 0.1 +  # Smoking History
                       (unlabeled_x[:, health_condition1_index] +
                        unlabeled_x[:, health_condition2_index] +
                        unlabeled_x[:, health_condition3_index] +
                        unlabeled_x[:, health_condition4_index] +
                        unlabeled_x[:, health_condition5_index]) * 0.1 +
                       unlabeled_x[:, arthritis_index] * 0.1)
        # Combine uncertainty and risk score
        combined_scores = (0.5 * uncertainties) + (0.5 * risk_scores)
        # Select samples with the highest combined score
        selected_indices = np.argsort(combined_scores)[-self.budget_per_iter:]
        return list(selected_indices)


    " Multi armed bandit pipline "

    def MAB_pipeline(self, mab, predicted_probabilities):

        def get_reward(pred_probs, real, epsilon=1e-3):
            return -math.log(pred_probs[real] + epsilon)

        mab = UCB(6)

        sampling_method = [self._random_sampling, self._uncertainty_sampling, self._diversity_sampling,
                           self._density_weighted_uncertainty_sampling, self._margin_sampling, self.risk_based_sampling]
        sm_rankings = {sm.__name__: sm(predicted_probabilities) for sm in sampling_method}
        chosen_samples = set()

        # 1 - initial step - choose each arm once
        for arm_idx, sm in enumerate(sampling_method):
            chosen_ind = sm_rankings[sm.__name__].pop(0)
            chosen_samples.add(chosen_ind)
            reward = get_reward(predicted_probabilities[chosen_ind], self.data['unlabeled_data']['y'][chosen_ind])
            mab.update(arm_idx, reward)

        # 2 - start exploration/exploitation
        while len(chosen_samples) != self.budget_per_iter:
            chosen_arm = mab.choose_arm()
            chosen_sample_ind = sm_rankings[sampling_method[chosen_arm].__name__].pop(0)
            chosen_samples.add(chosen_sample_ind)
            reward = get_reward(predicted_probabilities[chosen_sample_ind], self.data['unlabeled_data']['y'][chosen_sample_ind])
            mab.update(chosen_arm, reward)

        return list(chosen_samples)

    " Main Method "

    def run_pipeline(self, selection_criterion):
        """ Run the active learning pipeline """

        # labeled_data_x, labeled_data_y, unlabeled_data_x, unlabeled_data_y, test_data_x, test_data_y = \
        #     self.get_data_copy()

        self.copy_data = copy.deepcopy(self.data)


        mab = UCB(6)

        accuracy_scores = []
        for iteration in range(self.iterations):
            print(iteration)

            if len(self.copy_data['unlabeled_data']['y']) < self.budget_per_iter:
                break

            # 1. create and fit model on available train data
            train_x, train_y = self.copy_data['labeled_data']['x'], self.copy_data['labeled_data']['y']
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            # self.model = RandomForestClassifier(n_estimators=25, max_depth=5, random_state=42)
            self.model.fit(train_x, train_y)

            # 2. predict unlabeled data and choose samples to label (get probabilities, not predicted labels)
            unlabeled_x = self.copy_data['unlabeled_data']['x']
            y_pred = self.model.predict_proba(unlabeled_x)

            # 3. choose data to be labeled
            if selection_criterion == 'random':
                add_to_train_indices = self._random_sampling(y_pred)
            elif selection_criterion == 'uncertainty':
                add_to_train_indices = self._uncertainty_sampling(y_pred)
            # elif selection_criterion == 'worst_similarity':
            #     add_to_train_indices = self._worst_similarity_sampling(y_pred)

            elif selection_criterion == 'diversity':
                add_to_train_indices = self._diversity_sampling(y_pred)
            elif selection_criterion == 'density_weighted_uncertainty':
                add_to_train_indices = self._density_weighted_uncertainty_sampling(y_pred)
            elif selection_criterion == 'margin':
                add_to_train_indices = self._margin_sampling(y_pred)
            elif selection_criterion == 'risk_based':
                add_to_train_indices = self.risk_based_sampling(y_pred)

            elif selection_criterion == 'MAB':
                add_to_train_indices = self.MAB_pipeline(mab, y_pred)
            else:
                raise RuntimeError("unknown method")

            for idx in sorted(add_to_train_indices, reverse=True):
                self.copy_data['labeled_data']['x'].append(self.copy_data['unlabeled_data']['x'].pop(idx))
                self.copy_data['labeled_data']['y'] = np.append(self.copy_data['labeled_data']['y'], self.copy_data['unlabeled_data']['y'][idx])
                self.copy_data['unlabeled_data']['y'] = np.delete(self.copy_data['unlabeled_data']['y'], idx)

            # 4. Compute accuracy
            test_x, test_y = self.copy_data['test_data']['x'], self.copy_data['test_data']['y']
            y_pred = self.model.predict(test_x)
            accuracy = np.mean(y_pred == test_y)
            accuracy = round(float(accuracy), 3)
            accuracy_scores.append(accuracy)


        return accuracy_scores



if __name__ == '__main__':

    al = ActiveLearningPipeline(iterations=10, budget_per_iter=400, data_path=r"converted_data.csv",
                           train_label_test_split=(1000, 5000, 5500))

    # sampling_methods_to_try = ['MAB', 'random', 'uncertainty', 'worst_similarity']
    # sampling_methods_to_try = ['MAB', 'random', 'uncertainty']
    # sampling_methods_to_try = ['worst_similarity']

    # sampling_methods_to_try = ['diversity', 'density_weighted_uncertainty',
    #                            'margin', 'risk_based', 'random', 'uncertainty', 'MAB']

    sampling_methods_to_try = ['diversity', 'density_weighted_uncertainty',
                               'margin', 'risk_based', 'random', 'uncertainty', 'MAB']

    methods_performance = {}
    for sm in sampling_methods_to_try:
        print("="*10, sm, "="*10)
        sm_result = al.run_pipeline(selection_criterion=sm)
        methods_performance[sm] = sm_result

    generate_plot(methods_performance)
