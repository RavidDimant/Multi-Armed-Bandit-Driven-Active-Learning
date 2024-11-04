import copy
import math
import time
import warnings
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


def generate_plot(accuracy_scores_dict, title=''):
    """
    Generate a plot with specified figure size
    """
    plt.figure(figsize=(15, 9))  # Adjust width and height as desired
    for criterion, accuracy_scores in accuracy_scores_dict.items():
        plt.plot(range(1, len(accuracy_scores) + 1), accuracy_scores, label=criterion)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_sampling_results_bar(results):
    datasets = list(results.keys())  # List of dataset names
    sampling_methods = list(next(iter(results.values())).keys())  # List of sampling methods
    num_datasets = len(datasets)
    num_methods = len(sampling_methods)

    # Width of a single bar
    bar_width = 0.15
    # Generate an array of indices for the datasets
    x = np.arange(num_datasets)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(15, 9))

    # Plot each sampling method as a separate group of bars
    for i, method in enumerate(sampling_methods):
        # Get scores for this method across datasets
        scores = [results[dataset][method] for dataset in datasets]
        # Position bars with an offset for each sampling method
        ax.bar(x + i * bar_width, scores, width=bar_width, label=method)

    # Adding labels and title
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Score')
    ax.set_title('Sampling Method Results Across Datasets')
    ax.set_xticks(x + bar_width * (num_methods - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend(title='Sampling Method')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # Adjust layout for readability
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

    def __init__(self, feature_of_interest, iterations, budget_per_iter, data_path, train_label_test_split: tuple):

        self.model = None
        self.iterations = iterations
        self.feature_of_interest = feature_of_interest

        # read and prepare train data, data to label and test data
        data_df = pd.read_csv(data_path)
        data_df = shuffle(data_df, random_state=42)
        data_df = data_df.sample(frac=1).reset_index(drop=True)  # shuffle the df
        data_df = data_df.iloc[:2000]  ##########
        self.data_df = data_df

        print(f"\t-dataset size = {len(data_df)}")

        # Convert train_label_test_split from percentages to row indices
        total_rows = len(data_df)
        train_size = int(total_rows * train_label_test_split[0])
        label_size = int(total_rows * train_label_test_split[1])
        test_size = int(total_rows * train_label_test_split[2])
        if budget_per_iter == -1:
            self.budget_per_iter = int((label_size / self.iterations))
        else:
            self.budget_per_iter = budget_per_iter

        # Split data into train, label, and test sets
        labeled_df = data_df.iloc[:train_size]
        unlabeled_df = data_df.iloc[train_size:train_size + label_size]
        test_df = data_df.iloc[train_size + label_size:train_size + label_size + test_size]

        labeled_data = {'x': [np.array(row) for row in labeled_df.drop(feature_of_interest, axis=1).to_numpy()],
                      'y': np.array([l for l in labeled_df[feature_of_interest]])}

        unlabeled_data = {'x': [np.array(row) for row in unlabeled_df.drop(feature_of_interest, axis=1).to_numpy()],
                      'y':  np.array([l for l in unlabeled_df[feature_of_interest]])}

        test_data = {'x': [np.array(row) for row in test_df.drop(feature_of_interest, axis=1).to_numpy()],
                     'y':  np.array([l for l in test_df[feature_of_interest]])}

        self.data = {'labeled_data': labeled_data, 'unlabeled_data': unlabeled_data, 'test_data': test_data}
        self.copy_data = {}

        self.features = np.array(data_df.drop(feature_of_interest, axis=1).columns)
        self.sampling_methods = None

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
        return selected_indices

    def qbc_sampling(self, _):
        def train_committee(X, y, n_models):
            """
            Receives a dataset X and labels Y, sub-samples from them and trains `n_models` different logistic regression models.
            """
            # Create a list for the committe of models
            models = []
            total_samples = X.shape[0]
            n_samples = int(total_samples * 0.7)
            for _ in range(n_models):
                # Create a model
                model = LogisticRegression(max_iter=200)

                # Sample 70% of the dataset
                sampled_indices = np.random.choice(total_samples, size=n_samples, replace=False)
                X_sampled = X[sampled_indices]
                Y_sampled = y[sampled_indices]

                # Train the model
                model.fit(X_sampled, Y_sampled)

                # Save the trained model
                models.append(model)
            return models

        def qbc_disagreement(models, X_unlabeled):
            """
            Recieves a list of models and unlabeled data and via Query-By-Committee returns the entropy for all the predictions
            """
            n_models = len(models)
            # Create a variable to store the predictions of the committee
            predictions = np.zeros((n_models, X_unlabeled.shape[0]))  # shape: (n_models, n_samples)
            # Get the predicted label from each model in the committe
            for i, member in enumerate(models):
                predictions[i] = member.predict(X_unlabeled)
            # Tally the votes - for each label, count how many models classifed each sample as that label
            vote_counts = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int), minlength=2),
                axis=0,
                arr=predictions,
            )
            # Calculate the vote entropy as seen in the QBC formula
            vote_entropy = -np.sum((vote_counts / n_models) * np.log(vote_counts / n_models + 1e-10), axis=0)
            return vote_entropy

        n_select_per_iteration = self.budget_per_iter
        n_models = 7

        X_labeled = np.array(self.copy_data['labeled_data']['x'])
        y_labeled = np.array(self.copy_data['labeled_data']['y'])
        X_unlabeled = np.array(self.copy_data['unlabeled_data']['x'])

        # Train the committee of models
        models = train_committee(X_labeled, y_labeled, n_models)
        # Calculate disagreement
        disagreements = qbc_disagreement(models, X_unlabeled)
        # Select the samples with the highest disagreement
        selected_indices = np.argsort(disagreements)[-n_select_per_iteration:]
        return list(selected_indices)

    " Multi armed bandit pipline "

    def MAB_pipeline(self, mab, predicted_probabilities):

        # mab = UCB(len(self.sampling_methods))

        def get_reward(pred_probs, real, epsilon=1e-6):
            return -math.log(pred_probs[real] + epsilon)

        sm_rankings = {sm.__name__: sm(predicted_probabilities) for sm in self.sampling_methods}
        chosen_samples = set()

        # 1 - initial step - choose each arm once
        for arm_idx, sm in enumerate(self.sampling_methods):
            chosen_ind = sm_rankings[sm.__name__].pop(0)
            chosen_samples.add(chosen_ind)
            reward = get_reward(predicted_probabilities[chosen_ind], self.data['unlabeled_data']['y'][chosen_ind])
            mab.update(arm_idx, reward)

        # 2 - start exploration/exploitation
        while len(chosen_samples) != self.budget_per_iter:
            chosen_arm = mab.choose_arm()
            chosen_sample_ind = sm_rankings[self.sampling_methods[chosen_arm].__name__].pop(0)
            chosen_samples.add(chosen_sample_ind)
            reward = get_reward(predicted_probabilities[chosen_sample_ind], self.data['unlabeled_data']['y']
            [chosen_sample_ind])
            mab.update(chosen_arm, reward)

        return list(chosen_samples)

    " Main Method "

    def run_pipeline(self, selection_criterion):
        """ Run the active learning pipeline """

        self.copy_data = copy.deepcopy(self.data)

        self.sampling_methods = [self._random_sampling, self._uncertainty_sampling, self._diversity_sampling,
                           self._density_weighted_uncertainty_sampling, self._margin_sampling]
        mab = UCB(len(self.sampling_methods))


        accuracy_scores = []
        for iteration in range(self.iterations):

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
            elif selection_criterion == 'diversity':
                add_to_train_indices = self._diversity_sampling(y_pred)
            elif selection_criterion == 'density_weighted_uncertainty':
                add_to_train_indices = self._density_weighted_uncertainty_sampling(y_pred)
            elif selection_criterion == 'margin':
                add_to_train_indices = self._margin_sampling(y_pred)
            elif selection_criterion == 'QBC':
                add_to_train_indices = self.qbc_sampling(y_pred)
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


    datasets_names = ['car', 'diabetes', 'wine']  # , 'glass'
    label_per_data = ['unacc', 'Diabetes', 'quality']  # , 'Type'



    sampling_methods_to_try = ['random', 'uncertainty', 'diversity', 'density_weighted_uncertainty', 'margin', 'MAB']  # , 'QBC'

    dataset_performances = {dn: {sm: 0 for sm in sampling_methods_to_try} for dn in datasets_names}
    complete_history = {dn: [] for dn in datasets_names}

    for data_name, label in zip(datasets_names, label_per_data):
        print(f"Evaluating method on the {data_name} dataset.")

        cur_path = f"data/converted_{data_name}_data.csv"

        al = ActiveLearningPipeline(iterations=10, budget_per_iter=-1, data_path=cur_path,
                               train_label_test_split=(0.1, 0.8, 0.1), feature_of_interest=label)

        methods_performance = {}
        for sm in sampling_methods_to_try:
            print("="*10, sm, "="*10)
            sm_result = al.run_pipeline(selection_criterion=sm)
            methods_performance[sm] = sm_result

        for sm in sampling_methods_to_try:
            dataset_performances[data_name][sm] = np.mean(methods_performance[sm])  # mean over all iterations
            # dataset_performances[data_name][sm] = methods_performance[sm][-1]  # only final iteration result

        generate_plot(methods_performance, title=data_name.upper() + " DATASET")
        complete_history[data_name] = methods_performance


    plot_sampling_results_bar(dataset_performances)
    print(dataset_performances)
    # print("\n\n")
    # print(complete_history)

