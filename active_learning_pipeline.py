import copy
import math
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import entropy
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import random
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances

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


def plot_all_datasets_results(results):
    datasets = list(results.keys())  # List of dataset names
    sampling_methods = list(next(iter(results.values())).keys())  # List of sampling methods
    num_datasets = len(datasets)
    num_methods = len(sampling_methods)

    # Width of a single bar
    bar_width = 0.075
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
    ax.set_ylim(bottom=0.5)
    ax.set_title('Sampling Method Results Across Datasets')
    ax.set_xticks(x + bar_width * (num_methods - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend(title='Sampling Method')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # Adjust layout for readability
    plt.show()


def plot_bars_for_datasets(data_dict):
    # Loop through each dataset in the dictionary
    for dataset_name, dataset in data_dict.items():
        # Create a figure and axis for each dataset
        plt.figure(figsize=(10, 6))
        # Define the y-axis range (average ± 1)
        avg_value = np.mean(list(dataset.values()))
        plt.axhline(y=avg_value, color='red', linestyle='--', label=f'Average: {avg_value:.2f}')
        y_min = avg_value - 0.05
        y_max = avg_value + 0.05
        plt.ylim(y_min, y_max)

        # Plot the data as a bar chart
        categories = list(dataset.keys())
        values = list(dataset.values())
        plt.bar(categories, values, color='orange')

        # Add titles and labels
        plt.title(f'Bar Chart for {dataset_name} dataset')
        plt.xlabel('Categories')
        plt.ylabel('Values')
        plt.xticks(rotation=45, ha='right')
        # Display the plot
        plt.tight_layout()
        plt.show()


def plot_bars_for_methods(data_dict):
    # Prepare data for plotting by converting the dictionary to a flat list of records
    data = []
    for dataset_name, methods in data_dict.items():
        for method_name, score in methods.items():
            data.append({'Method': method_name, 'Dataset': dataset_name, 'Score': score})

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Set up the Seaborn plot
    plt.figure(figsize=(10, 6))
    sns.set_palette("Set2")  # You can adjust the color palette

    # Create the horizontal barplot with sub-bars for each method and dataset
    ax = sns.barplot(x='Score', y='Method', hue='Dataset', data=df, dodge=True, errorbar=None)

    # Add labels and title
    plt.title('Scores of Sampling Methods across Datasets')
    plt.xlabel('Score')
    plt.ylabel('Sampling Method')

    # Display the plot
    plt.tight_layout()
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
        data_df = data_df.iloc[:5000]  ##########
        self.data_df = data_df
        self.label_size = len(set(list(data_df[feature_of_interest])))

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
                          'y': np.array([l for l in unlabeled_df[feature_of_interest]])}

        test_data = {'x': [np.array(row) for row in test_df.drop(feature_of_interest, axis=1).to_numpy()],
                     'y': np.array([l for l in test_df[feature_of_interest]])}

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

    " Known Sampling Methods "

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
        # Select a subset of the data to limit memory usage
        subset_size = min(len(self.copy_data['unlabeled_data']['x']), 5000)

        if len(self.copy_data['unlabeled_data']['x']) > subset_size:
            indices = np.random.choice(len(self.copy_data['unlabeled_data']['x']), subset_size, replace=False)
            sampled_data = [self.copy_data['unlabeled_data']['x'][i] for i in indices]
        else:
            sampled_data = self.copy_data['unlabeled_data']['x']

        # Calculate pairwise distances between samples
        distance_matrix = pairwise_distances(np.array(sampled_data, dtype=np.float32))

        # Greedily select samples to maximize diversity
        selected_indices = []
        while len(selected_indices) < self.budget_per_iter:
            if not selected_indices:
                # Start with a random sample
                selected_indices.append(np.random.choice(range(subset_size)))
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
        subset_size = min(len(self.copy_data['unlabeled_data']['x']), 5000)

        if len(self.copy_data['unlabeled_data']['x']) > subset_size:
            indices = np.random.choice(len(self.copy_data['unlabeled_data']['x']), subset_size, replace=False)
            sampled_data = [self.copy_data['unlabeled_data']['x'][i] for i in indices]
            sampled_predicted_probabilities = self.model.predict_proba(sampled_data)
            uncertainties = entropy(sampled_predicted_probabilities.T)
        else:
            sampled_data = self.copy_data['unlabeled_data']['x']
        distances = pairwise_distances(np.array(sampled_data, dtype=np.float32))
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
            for _ in range(n_models):
                # Create a model
                model = LogisticRegression(max_iter=200)

                # Sample self.budget_per_iter of the dataset
                sampled_indices = np.random.choice(total_samples, size=self.budget_per_iter, replace=False)
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
            predictions = np.zeros((n_models, X_unlabeled.shape[0]))
            # Get the predicted label from each model in the committe
            for i, member in enumerate(models):
                predictions[i] = member.predict(X_unlabeled)
            # Tally the votes - for each label, count how many models classifed each sample as that label
            vote_counts = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int), minlength=self.label_size),
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

    " New Sampling Methods "

    def risk_based_sampling(self, y_pred):
        def get_top_correlated_features(top_n=10):
            # Compute correlations with feature_of_interest
            correlations = self.data_df.corr()[self.feature_of_interest].abs().sort_values(ascending=False)
            return np.array(correlations.index[1:top_n + 1])  # exclude feature_of_interest itself

        def assign_feature_weights(correlated_features):
            # Define a weight dictionary, modify as needed for different datasets
            weights = {feature: 1 / len(correlated_features) for feature in
                       correlated_features}  # Equal weight for simplicity
            return weights

        uncertainties = entropy(y_pred.T)
        num_of_features = len(self.features)
        correlated_features = get_top_correlated_features(top_n=int(0.3 * num_of_features))
        feature_indices = [list(self.features).index(f) for f in correlated_features]

        unlabeled_x = np.array(self.copy_data['unlabeled_data']['x'])
        risk_scores = np.zeros(len(unlabeled_x))
        # weights based on feature correlations or importance
        weights = assign_feature_weights(correlated_features)  # Assumes a dictionary with feature weights

        for i, feature_index in enumerate(feature_indices):
            # Rescale feature to 0-1 if it isn’t already
            feature_values = unlabeled_x[:, feature_index]
            # feature_min, feature_max = feature_values.min(), feature_values.max()
            # normalized_feature = (feature_values - feature_min) / (feature_max - feature_min)
            # Update risk scores with weighted normalized values
            risk_scores += feature_values * weights[correlated_features[i]]

        # Combine uncertainty and risk score
        combined_scores = (0.5 * uncertainties) + (0.5 * risk_scores)
        # Select samples with the highest combined score
        selected_indices = np.argsort(combined_scores)[-self.budget_per_iter:]
        return list(selected_indices)

    def metropolis_hastings_sampling(self, predicted_probabilities):
        if len(predicted_probabilities.shape) > 1:
            predicted_probabilities = np.max(predicted_probabilities, axis=1)
        proposal_std = 0.1
        num_data = len(predicted_probabilities)
        selected_indices = []
        # Start with a random initial index
        current_index = np.random.randint(0, num_data)
        selected_indices.append(current_index)
        while len(selected_indices) < self.budget_per_iter:
            # Propose a new index by adding Gaussian noise to the current index
            proposed_index = int(current_index + np.random.normal(0, proposal_std) * num_data)
            proposed_index = np.clip(proposed_index, 0, num_data - 1)

            # Acceptance criterion: higher probabilities are preferred
            acceptance_ratio = min(
                1,
                predicted_probabilities[proposed_index] / predicted_probabilities[current_index]
            )
            # Accept or reject the proposal
            if np.random.rand() < acceptance_ratio:
                selected_indices.append(proposed_index)
                current_index = proposed_index
        return list(set(selected_indices))

    def thompson_sampling(self, predicted_probabilities):
        y_true = self.copy_data['unlabeled_data']['y']
        num_data = len(predicted_probabilities)
        selected_indices = []

        # Initialize success/failure counts for each sample
        successes = np.ones(num_data)  # Start with prior of 1 success
        failures = np.ones(num_data)  # Start with prior of 1 failure

        for _ in range(self.budget_per_iter):
            # Draw a sample for each index from the Beta distribution
            samples = np.random.beta(successes + 1, failures + 1)
            # Select the index with the highest sampled probability
            chosen_index = np.argmax(samples)
            selected_indices.append(chosen_index)
            # Update success/failure counts based on actual reward
            if y_true[chosen_index] == 1:
                successes[chosen_index] += 1
            else:
                failures[chosen_index] += 1
        return list(set(selected_indices))

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

            if len(sm_rankings[self.sampling_methods[chosen_arm].__name__]) == 0:
                break

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
                                 self._density_weighted_uncertainty_sampling, self._margin_sampling, self.qbc_sampling,
                                 self.risk_based_sampling, self.metropolis_hastings_sampling, self.thompson_sampling]

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
            elif selection_criterion == 'risk_based':
                add_to_train_indices = self.risk_based_sampling(y_pred)
            elif selection_criterion == 'metropolis_hastings':
                add_to_train_indices = self.metropolis_hastings_sampling(y_pred)
            elif selection_criterion == 'thompson':
                add_to_train_indices = self.thompson_sampling(y_pred)
            else:
                raise RuntimeError("unknown method")

            for idx in sorted(add_to_train_indices, reverse=True):
                self.copy_data['labeled_data']['x'].append(self.copy_data['unlabeled_data']['x'].pop(idx))
                self.copy_data['labeled_data']['y'] = np.append(self.copy_data['labeled_data']['y'],
                                                                self.copy_data['unlabeled_data']['y'][idx])
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

    sampling_methods_to_try = ['random', 'uncertainty', 'diversity', 'density_weighted_uncertainty', 'margin', 'QBC',
                               'risk_based', 'metropolis_hastings', 'thompson', 'MAB']

    dataset_performances = {dn: {sm: 0 for sm in sampling_methods_to_try} for dn in datasets_names}
    complete_history = {dn: [] for dn in datasets_names}

    for data_name, label in zip(datasets_names, label_per_data):
        print(f"Evaluating method on the {data_name} dataset.")

        cur_path = f"data/converted_{data_name}_data.csv"

        al = ActiveLearningPipeline(iterations=10, budget_per_iter=-1, data_path=cur_path,
                                    train_label_test_split=(0.1, 0.8, 0.1), feature_of_interest=label)

        methods_performance = {}
        for sm in sampling_methods_to_try:
            print("=" * 10, sm, "=" * 10)
            sm_result = al.run_pipeline(selection_criterion=sm)
            methods_performance[sm] = sm_result

        for sm in sampling_methods_to_try:
            dataset_performances[data_name][sm] = np.mean(methods_performance[sm])  # mean over all iterations
            # dataset_performances[data_name][sm] = methods_performance[sm][-1]  # only final iteration result

        generate_plot(methods_performance, title=data_name.upper() + " DATASET")
        complete_history[data_name] = methods_performance

    plot_all_datasets_results(dataset_performances)
    plot_bars_for_datasets(dataset_performances)
    plot_bars_for_methods(dataset_performances)
    print(dataset_performances)
