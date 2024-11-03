import numpy as np
import pandas as pd
import random
import math
import copy
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

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

        # Read and prepare train, unlabeled, and test data
        data_df = pd.read_csv(data_path)
        data_df = shuffle(data_df, random_state=42)

        labeled_df = data_df.iloc[:train_label_test_split[0]]
        unlabeled_df = data_df.iloc[train_label_test_split[0]:train_label_test_split[1]]
        test_df = data_df.iloc[train_label_test_split[1]:train_label_test_split[2]]

        labeled_data = {'x': [np.array(row) for row in labeled_df.drop('diabetes', axis=1).to_numpy()],
                        'y': np.array([l for l in labeled_df['diabetes']])}

        unlabeled_data = {'x': [np.array(row) for row in unlabeled_df.drop('diabetes', axis=1).to_numpy()],
                          'y': np.array([l for l in unlabeled_df['diabetes']])}

        test_data = {'x': [np.array(row) for row in test_df.drop('diabetes', axis=1).to_numpy()],
                     'y': np.array([l for l in test_df['diabetes']])}

        self.data = {'labeled_data': labeled_data, 'unlabeled_data': unlabeled_data, 'test_data': test_data}
        self.copy_data = {}
        self.features = np.array(data_df.drop('diabetes', axis=1).columns)

    def risk_based_sampling(self, y_pred):
        uncertainties = entropy(y_pred.T)

        # Adjusted risk factors for this dataset
        features_list = list(self.features)
        age_index = features_list.index('age')
        blood_pressure_index = features_list.index('high_blood_pressure')
        creatinine_index = features_list.index('serum_creatinine')
        sodium_index = features_list.index('serum_sodium')
        smoking_index = features_list.index('smoking')

        unlabeled_x = np.array(self.copy_data['unlabeled_data']['x'])
        age_normalized = (unlabeled_x[:, age_index] - 40) / (95 - 40)  # Normalizing age
        risk_scores = (age_normalized * 0.25 +
                       unlabeled_x[:, blood_pressure_index] * 0.2 +
                       unlabeled_x[:, creatinine_index] * 0.3 +
                       unlabeled_x[:, sodium_index] * -0.15 +
                       unlabeled_x[:, smoking_index] * 0.1)

        combined_scores = (0.5 * uncertainties) + (0.5 * risk_scores)
        selected_indices = np.argsort(combined_scores)[-self.budget_per_iter:]
        return list(selected_indices)

    def _random_sampling(self, predicted_probabilities):
        pool_size = len(predicted_probabilities)
        n_select = min(self.budget_per_iter, pool_size)
        selected_indices = random.sample(range(pool_size), n_select)
        return selected_indices

    def _uncertainty_sampling(self, predicted_probabilities):
        uncertainties = entropy(predicted_probabilities.T)
        selected_indices = np.argsort(uncertainties)[-self.budget_per_iter:].astype(int).tolist()
        return selected_indices

    def qbc_sampling(self, _):
        def train_committee(X, y, n_models=7):
            models = []
            total_samples = X.shape[0]
            n_samples = int(total_samples * 0.7)
            for _ in range(n_models):
                model = LogisticRegression(max_iter=200)
                sampled_indices = np.random.choice(total_samples, size=n_samples, replace=False)
                X_sampled = X[sampled_indices]
                Y_sampled = y[sampled_indices]
                model.fit(X_sampled, Y_sampled)
                models.append(model)
            return models

        def qbc_disagreement(models, X_unlabeled):
            n_models = len(models)
            predictions = np.zeros((n_models, X_unlabeled.shape[0]))
            for i, member in enumerate(models):
                predictions[i] = member.predict(X_unlabeled)
            vote_counts = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int), minlength=2),
                axis=0,
                arr=predictions,
            )
            vote_entropy = -np.sum((vote_counts / n_models) * np.log(vote_counts / n_models + 1e-10), axis=0)
            return vote_entropy

        X_labeled = np.array(self.copy_data['labeled_data']['x'])
        y_labeled = np.array(self.copy_data['labeled_data']['y'])
        X_unlabeled = np.array(self.copy_data['unlabeled_data']['x'])

        models = train_committee(X_labeled, y_labeled)
        disagreements = qbc_disagreement(models, X_unlabeled)
        selected_indices = np.argsort(disagreements)[-self.budget_per_iter:]
        return selected_indices

    def MAB_pipeline(self, mab, predicted_probabilities):
        def get_reward(pred_probs, real, epsilon=1e-3):
            return -math.log(pred_probs[real] + epsilon)

        sampling_methods = [self._random_sampling, self._uncertainty_sampling, self.risk_based_sampling,
                            self.qbc_sampling]
        sm_rankings = {sm.__name__: list(sm(predicted_probabilities)) for sm in sampling_methods}
        chosen_samples = set()

        # 1 - initial step - choose each arm once
        for arm_idx, sm in enumerate(sampling_methods):
            chosen_ind = sm_rankings[sm.__name__].pop(0)
            chosen_samples.add(chosen_ind)
            reward = get_reward(predicted_probabilities[chosen_ind], self.data['unlabeled_data']['y'][chosen_ind])
            mab.update(arm_idx, reward)

        # 2 - start exploration/exploitation
        while len(chosen_samples) < self.budget_per_iter:
            chosen_arm = mab.choose_arm()
            chosen_sample_ind = sm_rankings[sampling_methods[chosen_arm].__name__].pop(0)
            chosen_samples.add(chosen_sample_ind)
            reward = get_reward(predicted_probabilities[chosen_sample_ind],
                                self.data['unlabeled_data']['y'][chosen_sample_ind])
            mab.update(chosen_arm, reward)

        return list(chosen_samples)

    def run_pipeline(self, selection_criterion):
        self.copy_data = copy.deepcopy(self.data)
        accuracy_scores = []
        mab = UCB(4)
        for iteration in range(self.iterations):
            if len(self.copy_data['unlabeled_data']['y']) < self.budget_per_iter:
                break

            train_x, train_y = self.copy_data['labeled_data']['x'], self.copy_data['labeled_data']['y']
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model.fit(train_x, train_y)

            unlabeled_x = self.copy_data['unlabeled_data']['x']
            y_pred = self.model.predict_proba(unlabeled_x)

            if selection_criterion == 'random':
                add_to_train_indices = self._random_sampling(y_pred)
            elif selection_criterion == 'risk_based':
                add_to_train_indices = self.risk_based_sampling(y_pred)
            elif selection_criterion == 'uncertainty':
                add_to_train_indices = self._uncertainty_sampling(y_pred)
            elif selection_criterion == 'QBC':
                add_to_train_indices = self.qbc_sampling(y_pred)
            elif selection_criterion == 'MAB':
                add_to_train_indices = self.MAB_pipeline(mab, y_pred)
            else:
                raise RuntimeError("unknown method")

            for idx in sorted(add_to_train_indices, reverse=True):
                self.copy_data['labeled_data']['x'].append(self.copy_data['unlabeled_data']['x'].pop(idx))
                self.copy_data['labeled_data']['y'] = np.append(self.copy_data['labeled_data']['y'],
                                                                self.copy_data['unlabeled_data']['y'][idx])
                self.copy_data['unlabeled_data']['y'] = np.delete(self.copy_data['unlabeled_data']['y'], idx)

            test_x, test_y = self.copy_data['test_data']['x'], self.copy_data['test_data']['y']
            y_pred = self.model.predict(test_x)
            accuracy = np.mean(y_pred == test_y)
            accuracy_scores.append(round(float(accuracy), 3))

        return accuracy_scores

# Run the pipeline with multiple sampling methods
data_path = 'heart_failure_clinical_records_dataset.csv'
al_pipeline = ActiveLearningPipeline(iterations=10, budget_per_iter=30, data_path=data_path, train_label_test_split=(100, 200, 250))
methods_performance = {}

# Define sampling methods to test
sampling_methods_to_try = ['MAB', 'QBC', 'risk_based', 'random', 'uncertainty']

# Testing with the specified sampling methods
for sm in sampling_methods_to_try:
    print("=" * 10, sm, "=" * 10)
    sm_result = al_pipeline.run_pipeline(selection_criterion=sm)
    methods_performance[sm] = sm_result

# Plotting the performance
def generate_plot(accuracy_scores_dict):
    for criterion, accuracy_scores in accuracy_scores_dict.items():
        plt.plot(range(1, len(accuracy_scores) + 1), accuracy_scores, label=criterion)

    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Scores for Different Sampling Criteria')
    plt.legend()
    plt.grid(True)
    plt.show()

generate_plot(methods_performance)
