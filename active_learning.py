import copy
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import entropy
from sklearn.utils import shuffle
import random
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from multi_arm_bandit import MAB

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class ActiveLearning:

    def __init__(self, data_path, feature_of_interest, size_limit, iterations, budget_per_iter,
                 train_label_test_split: tuple):

        # read data
        data_df = pd.read_csv(data_path)
        data_df = shuffle(data_df)  # , random_state=42
        data_df = data_df.iloc[:size_limit]

        # convert train_label_test_split from percentages to row indices
        total_rows = len(data_df)
        train_size = int(total_rows * train_label_test_split[0])
        label_size = int(total_rows * train_label_test_split[1])
        test_size = int(total_rows * train_label_test_split[2])

        # set data attributes
        self.data_df = data_df
        self.features = np.array(data_df.drop(feature_of_interest, axis=1).columns)
        self.feature_of_interest = feature_of_interest
        self.label_size = len(set(list(data_df[feature_of_interest])))

        # set training parameters
        self.iterations = iterations
        if budget_per_iter == 0:  # enough budget to label all the unlabeled data
            self.budget_per_iter = int((label_size / self.iterations))
        elif budget_per_iter == -1:  # enough budget to label only 2/3 of the unlabeled data
            self.budget_per_iter = int(((label_size * (2/3)) / self.iterations))  # can only label x% of possible data
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
        self.data_copy = {}


        self.model = None

        self.sampling_methods = self.SamplingMethods(self)
        self.sampling_pipelines = self.SamplingPipelines(self)

    " Auxiliary Methods "

    def get_data_copy(self):
        labeled_data_x = copy.deepcopy(self.data['labeled_data']['x'])
        labeled_data_y = copy.deepcopy(self.data['labeled_data']['y'])

        unlabeled_data_x = copy.deepcopy(self.data['unlabeled_data']['x'])
        unlabeled_data_y = copy.deepcopy(self.data['unlabeled_data']['y'])

        test_data_x = copy.deepcopy(self.data['test_data']['x'])
        test_data_y = copy.deepcopy(self.data['test_data']['y'])

        return labeled_data_x, labeled_data_y, unlabeled_data_x, unlabeled_data_y, test_data_x, test_data_y

    " Sampling Methods & Pipelines "

    class SamplingMethods:
        """ sampling methods class"""

        def __init__(self, parent):
            self.parent = parent  # Reference to the main class instance

        " Known Sampling Methods "

        def random_sampling(self, predicted_probabilities):
            pool_size = len(predicted_probabilities)
            to_label_size = pool_size
            n_select = min(self.parent.budget_per_iter, to_label_size)
            # Randomly select n_select indices without replacement
            selected_indices = random.sample(range(to_label_size), n_select)

            return selected_indices

        def uncertainty_sampling(self, predicted_probabilities):
            to_label_size = len(predicted_probabilities)
            n_select = min(self.parent.budget_per_iter, to_label_size)
            uncertainties = entropy(predicted_probabilities.T)
            selected_indices = np.argsort(uncertainties)[-n_select:].astype(int).tolist()
            selected_indices = [i for i in reversed(selected_indices)]

            return selected_indices

        def diversity_sampling(self, _):
            # Select a subset of the data to limit memory usage
            subset_size = min(len(self.parent.data_copy['unlabeled_data']['x']), 1500)

            if len(self.parent.data_copy['unlabeled_data']['x']) > subset_size:
                indices = np.random.choice(len(self.parent.data_copy['unlabeled_data']['x']), subset_size, replace=False)
                sampled_data = [self.parent.data_copy['unlabeled_data']['x'][i] for i in indices]
            else:
                sampled_data = self.parent.data_copy['unlabeled_data']['x']

            # Calculate pairwise distances between samples
            distance_matrix = pairwise_distances(np.array(sampled_data, dtype=np.float32))

            # Greedily select samples to maximize diversity
            selected_indices = []
            while len(selected_indices) < self.parent.budget_per_iter:
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

        def density_weighted_uncertainty_sampling(self, predicted_probabilities):
            uncertainties = entropy(predicted_probabilities.T)
            # Calculate sample density in the feature space
            subset_size = min(len(self.parent.data_copy['unlabeled_data']['x']), 1500)

            if len(self.parent.data_copy['unlabeled_data']['x']) > subset_size:
                indices = np.random.choice(len(self.parent.data_copy['unlabeled_data']['x']), subset_size, replace=False)
                sampled_data = [self.parent.data_copy['unlabeled_data']['x'][i] for i in indices]
                sampled_predicted_probabilities = self.parent.model.predict_proba(sampled_data)
                uncertainties = entropy(sampled_predicted_probabilities.T)
            else:
                sampled_data = self.parent.data_copy['unlabeled_data']['x']
            distances = pairwise_distances(np.array(sampled_data, dtype=np.float32))
            densities = np.sum(np.exp(-distances), axis=1)  # Higher density for closer points

            # Combine uncertainty and density
            combined_scores = uncertainties * densities
            selected_indices = list(np.argsort(combined_scores)[-self.parent.budget_per_iter:])
            return selected_indices

        def margin_sampling(self, predicted_probabilities):
            # Calculate the margin between the top two probabilities
            sorted_probs = np.sort(predicted_probabilities, axis=1)
            margins = sorted_probs[:, -1] - sorted_probs[:, -2]
            # Select samples with the smallest margins (closest to the decision boundary)
            selected_indices = list(np.argsort(margins)[:self.parent.budget_per_iter])
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
                    sampled_indices = np.random.choice(total_samples, size=self.parent.budget_per_iter, replace=False)
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
                    lambda x: np.bincount(x.astype(int), minlength=self.parent.label_size),
                    axis=0,
                    arr=predictions,
                )
                # Calculate the vote entropy as seen in the QBC formula
                vote_entropy = -np.sum((vote_counts / n_models) * np.log(vote_counts / n_models + 1e-10), axis=0)
                return vote_entropy

            n_select_per_iteration = self.parent.budget_per_iter
            n_models = 7

            X_labeled = np.array(self.parent.data_copy['labeled_data']['x'])
            y_labeled = np.array(self.parent.data_copy['labeled_data']['y'])
            X_unlabeled = np.array(self.parent.data_copy['unlabeled_data']['x'])

            # Train the committee of models
            models = train_committee(X_labeled, y_labeled, n_models)
            # Calculate disagreement
            disagreements = qbc_disagreement(models, X_unlabeled)
            # Select the samples with the highest disagreement
            selected_indices = np.argsort(disagreements)[-n_select_per_iteration:]
            return list(selected_indices)

        def metropolis_hastings_sampling(self, predicted_probabilities):
            if len(predicted_probabilities.shape) > 1:
                predicted_probabilities = np.max(predicted_probabilities, axis=1)
            proposal_std = 0.1
            num_data = len(predicted_probabilities)
            selected_indices = set()
            # Start with a random initial index
            current_index = np.random.randint(0, num_data)
            selected_indices.add(current_index)
            while len(selected_indices) < self.parent.budget_per_iter:
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
                    selected_indices.add(proposed_index)
                    current_index = proposed_index

            return list(selected_indices)

        " New Sampling Method "

        def feature_based_sampling(self, predicted_probabilities):
            def get_top_correlated_features(top_n=10):
                # Compute correlations with feature_of_interest
                correlations = self.parent.data_df.corr()[self.parent.feature_of_interest].abs().sort_values(ascending=False)
                return np.array(correlations.index[1:top_n + 1])  # exclude feature_of_interest itself

            def assign_feature_weights(correlated_features):
                # Define a weight dictionary, modify as needed for different datasets
                weights = {feature: 1 / len(correlated_features) for feature in
                           correlated_features}  # Equal weight for simplicity
                return weights

            uncertainties = entropy(predicted_probabilities.T)
            num_of_features = len(self.parent.features)
            correlated_features = get_top_correlated_features(top_n=int(0.3 * num_of_features))
            feature_indices = [list(self.parent.features).index(f) for f in correlated_features]

            unlabeled_x = np.array(self.parent.data_copy['unlabeled_data']['x'])
            risk_scores = np.zeros(len(unlabeled_x))
            # weights based on feature correlations or importance
            weights = assign_feature_weights(correlated_features)  # Assumes a dictionary with feature weights

            for i, feature_index in enumerate(feature_indices):
                # Rescale feature to 0-1 if it isn’t already
                feature_values = unlabeled_x[:, feature_index]
                # Update risk scores with weighted normalized values
                risk_scores += feature_values * weights[correlated_features[i]]

            # Combine uncertainty and risk score
            combined_scores = (0.5 * uncertainties) + (0.5 * risk_scores)
            # Select samples with the highest combined score
            selected_indices = np.argsort(combined_scores)[-self.parent.budget_per_iter:]
            return list(selected_indices)

    class SamplingPipelines:
        """ sampling pipelines - uses pool of sampling methods to select samples to label """

        def __init__(self, parent):
            self.parent = parent  # Reference to the main class instance
            self.all_sm = []
            # [self.parent.sampling_methods.random_sampling,
            #                self.parent.sampling_methods.uncertainty_sampling,
            #                self.parent.sampling_methods.diversity_sampling,
            #                self.parent.sampling_methods.density_weighted_uncertainty_sampling,
            #                self.parent.sampling_methods.margin_sampling,
            #                self.parent.sampling_methods.qbc_sampling,
            #                self.parent.sampling_methods.metropolis_hastings_sampling,
            #                # self.parent.sampling_methods.thompson_sampling,
            #                self.parent.sampling_methods.feature_based_sampling]
            self.vanilla_mab = None
            self.lst_mab = None

        def random_all_sampling(self, predicted_probabilities):
            """ choose each sample to label by a new random sampling method """
            sm_rankings = {sm.__name__: sm(predicted_probabilities) for sm in self.all_sm}
            chosen_samples = set()
            while len(chosen_samples) != self.parent.budget_per_iter:
                # draw sampling methods
                cur_random_sm = random.randint(0, len(self.all_sm) - 1)
                # choose sample to label (repeat until un-chosen sample is chosen)
                while True:
                    if len(sm_rankings[self.all_sm[cur_random_sm].__name__]) == 0:
                        return list(chosen_samples)
                    chosen_sample_ind = sm_rankings[self.all_sm[cur_random_sm].__name__].pop(0)
                    if chosen_sample_ind not in chosen_samples:
                        chosen_samples.add(chosen_sample_ind)
                        break

            return list(chosen_samples)

        def vanilla_MAB(self, predicted_probabilities):
            """
            A single multi armed bandit for all iterations
            :param predicted_probabilities: probabilities given to the samples that can be labeled
            :return: indices of chosen sampled to label
            """

            if self.vanilla_mab is None:
                self.vanilla_mab = MAB(len(self.all_sm))

            sm_rankings = {sm.__name__: sm(predicted_probabilities) for sm in self.all_sm}
            chosen_samples = set()

            # 1 - initial step - choose each arm once
            if self.vanilla_mab.initialized is False:
                for arm_idx, sm in enumerate(self.all_sm):
                    chosen_ind = sm_rankings[sm.__name__].pop(0)
                    chosen_samples.add(chosen_ind)
                    reward = self.vanilla_mab.vx_reward(predicted_probabilities[chosen_ind],
                                                        self.parent.data['unlabeled_data']['y'][chosen_ind])
                    self.vanilla_mab.update(arm_idx, reward)
                self.vanilla_mab.initialized = True

            # 2 - start exploration/exploitation
            while len(chosen_samples) != self.parent.budget_per_iter:
                chosen_arm = self.vanilla_mab.ucb_choose_arm()
                # if current arm has not more samples that can be labeled, then all arms don't have samples
                if len(sm_rankings[self.all_sm[chosen_arm].__name__]) == 0:
                    break
                chosen_sample_ind = sm_rankings[self.all_sm[chosen_arm].__name__].pop(0)
                chosen_samples.add(chosen_sample_ind)

                reward = self.vanilla_mab.vx_reward(predicted_probabilities[chosen_sample_ind],
                                                    self.parent.data['unlabeled_data']['y']
                                              [chosen_sample_ind])
                self.vanilla_mab.update(chosen_arm, reward)

            return list(chosen_samples)

        def lst_MAB(self, predicted_probabilities):

            def choose_mab(old_ucb, new_ucb):
                kl_divergence = lambda p, q: np.sum(p * np.log(p / (q + 1e-10)))

                old_weights = np.array([v / sum(old_ucb.mus) for v in old_ucb.mus])
                new_weights = np.array([v / sum(new_ucb.mus) for v in new_ucb.mus])

                kl_value = kl_divergence(old_weights, new_weights)

                if kl_value > 0.1:
                    return 'short-term'
                return 'long-term'


            if self.lst_mab is None:
                self.lst_mab = MAB(len(self.all_sm))

            sm_rankings = {sm.__name__: sm(predicted_probabilities) for sm in self.all_sm}
            chosen_samples = set()

            def initialize_mab(a_mab):
                """ choose each possible arm to get initial reward estimation """
                for arm_idx, sm in enumerate(self.all_sm):
                    chosen_ind = sm_rankings[sm.__name__].pop(0)
                    chosen_samples.add(chosen_ind)
                    reward = a_mab.vx_reward(predicted_probabilities[chosen_ind],
                                             self.parent.data['unlabeled_data']['y'][chosen_ind])
                    a_mab.update(arm_idx, reward)
                a_mab.initialized = True

            def explore_exploits_mab(a_mab, budget):
                while len(chosen_samples) != budget:
                    chosen_arm = a_mab.ucb_choose_arm()
                    if len(sm_rankings[self.all_sm[chosen_arm].__name__]) == 0:
                        break
                    chosen_sample_ind = sm_rankings[self.all_sm[chosen_arm].__name__].pop(0)
                    chosen_samples.add(chosen_sample_ind)
                    reward = a_mab.vx_reward(predicted_probabilities[chosen_sample_ind],
                                             self.parent.data['unlabeled_data']['y']
                                               [chosen_sample_ind])
                    a_mab.update(chosen_arm, reward)

            short_term_mab = MAB(len(self.all_sm))
            long_term_mab = self.lst_mab

            # on first iteration, initalize and use and existing (empty) MAB
            on_first_iteration = long_term_mab.initialized is False
            if on_first_iteration:
                initialize_mab(long_term_mab)
                explore_exploits_mab(long_term_mab, budget=self.parent.budget_per_iter)

            # on other iterations, train a new MAB, compare long-term and short-term MAB, and label choose labels
            else:
                initialize_mab(short_term_mab)
                explore_exploits_mab(short_term_mab, budget=int(0.1 * self.parent.budget_per_iter))
                mab_to_use = choose_mab(long_term_mab, short_term_mab)
                if mab_to_use == 'short-term':
                    mab_to_use = short_term_mab
                    self.lst_mab = short_term_mab
                else:
                    mab_to_use = long_term_mab

                explore_exploits_mab(mab_to_use, budget=self.parent.budget_per_iter)

            return list(chosen_samples)

    " Main Methods "

    def run_sm(self, method, random_seed=42):
        """ run Active-Learning pipeline with a given method (either sampling method or sampling pipeline)"""

        self.data_copy = copy.deepcopy(self.data)

        accuracy_scores = []
        for iteration in range(self.iterations):

            if len(self.data_copy['unlabeled_data']['y']) < self.budget_per_iter:
                break

            # 1. create and fit model on available train data
            train_x, train_y = self.data_copy['labeled_data']['x'], self.data_copy['labeled_data']['y']
            self.model = RandomForestClassifier(n_estimators=25, max_depth=5, random_state=random_seed)
            self.model.fit(train_x, train_y)

            # 2. predict unlabeled data and choose samples to label (get probabilities, not predicted labels)
            unlabeled_x = self.data_copy['unlabeled_data']['x']
            y_pred = self.model.predict_proba(unlabeled_x)

            # 3. choose data to be labeled & label it
            add_to_train_indices = method(y_pred)

            for idx in sorted(add_to_train_indices, reverse=True):
                self.data_copy['labeled_data']['x'].append(self.data_copy['unlabeled_data']['x'].pop(idx))
                self.data_copy['labeled_data']['y'] = np.append(self.data_copy['labeled_data']['y'],
                                                                self.data_copy['unlabeled_data']['y'][idx])
                self.data_copy['unlabeled_data']['y'] = np.delete(self.data_copy['unlabeled_data']['y'], idx)

            # 4. Compute and save current iteration test accuracy
            test_x, test_y = self.data_copy['test_data']['x'], self.data_copy['test_data']['y']
            y_pred = self.model.predict(test_x)
            accuracy = np.mean(y_pred == test_y)
            accuracy = round(float(accuracy), 3)
            accuracy_scores.append(accuracy)

        return accuracy_scores

    def run_experiments(self, methods2use=('all', )):

        random_seed = random.randint(1, 42)

        # set methods to use and method pool for MABS
        sampling_methods_to_test = []
        if 'random_sampling' in methods2use or 'all' in methods2use:
            sampling_methods_to_test.append(self.sampling_methods.random_sampling)
            self.sampling_pipelines.all_sm.append(self.sampling_methods.random_sampling)
        if 'uncertainty_sampling' in methods2use or 'all' in methods2use:
            sampling_methods_to_test.append(self.sampling_methods.uncertainty_sampling)
            self.sampling_pipelines.all_sm.append(self.sampling_methods.uncertainty_sampling)
        if 'diversity_sampling' in methods2use or 'all' in methods2use:
            sampling_methods_to_test.append(self.sampling_methods.diversity_sampling)
            self.sampling_pipelines.all_sm.append(self.sampling_methods.diversity_sampling)
        if 'density_weighted_uncertainty_sampling' in methods2use or 'all' in methods2use:
            sampling_methods_to_test.append(self.sampling_methods.density_weighted_uncertainty_sampling)
            self.sampling_pipelines.all_sm.append(self.sampling_methods.density_weighted_uncertainty_sampling)
        if 'margin_sampling' in methods2use or 'all' in methods2use:
            sampling_methods_to_test.append(self.sampling_methods.margin_sampling)
            self.sampling_pipelines.all_sm.append(self.sampling_methods.margin_sampling)
        if 'qbc_sampling' in methods2use or 'all' in methods2use:
            sampling_methods_to_test.append(self.sampling_methods.qbc_sampling)
            self.sampling_pipelines.all_sm.append(self.sampling_methods.qbc_sampling)
        if 'metropolis_hastings_sampling' in methods2use or 'all' in methods2use:
            sampling_methods_to_test.append(self.sampling_methods.metropolis_hastings_sampling)
            self.sampling_pipelines.all_sm.append(self.sampling_methods.metropolis_hastings_sampling)
        if 'feature_based_sampling' in methods2use or 'all' in methods2use:
            sampling_methods_to_test.append(self.sampling_methods.feature_based_sampling)
            self.sampling_pipelines.all_sm.append(self.sampling_methods.feature_based_sampling)
        if 'random_all_sampling' in methods2use or 'all' in methods2use:
            sampling_methods_to_test.append(self.sampling_pipelines.random_all_sampling)
        if 'vanilla_MAB' in methods2use or 'all' in methods2use:
            sampling_methods_to_test.append(self.sampling_pipelines.vanilla_MAB)
        if 'lst_MAB' in methods2use or 'all' in methods2use:
            sampling_methods_to_test.append(self.sampling_pipelines.lst_MAB)

        # for each method, run Active-Learning pipeline and save results
        methods_performance = {}
        for sm in sampling_methods_to_test:
            methods_performance[sm.__name__] = {}

            sm_result = self.run_sm(method=sm, random_seed=random_seed)

            methods_performance[sm.__name__]['mean'] = np.mean(sm_result)
            methods_performance[sm.__name__]['last'] = sm_result[-1]

        return methods_performance



if __name__ == '__main__':

    datasets_pools = ['apple', 'loan', 'mb', 'passenger', 'diabetes', 'employee', 'shipping', 'hotel']
    label_per_data = ['Quality', 'loan_status', 'Preference', 'satisfaction', 'Diabetes', 'LeaveOrNot',
                      'Reached.on.Time_Y.N', 'booking_status']
    sampling_methods_pool = ['random_sampling', 'uncertainty_sampling', 'diversity_sampling',
                             'density_weighted_uncertainty_sampling', 'margin_sampling', 'qbc_sampling',
                             'metropolis_hastings_sampling', 'thompson_sampling', 'feature_based_sampling',
                             'random_all_sampling', 'vanilla_MAB', 'lst_MAB']


    # specify datasets names to use for the evaluations
    dataset_to_use = datasets_pools
    # specify methods names to use for the evaluations
    methods_to_use = ("all", )


    # perform bootstrap
    bootstrap_size = 100
    bootstrap_results = {d: [] for d in dataset_to_use}
    # for dn, l in zip(dataset_to_use, label_per_data):
    for i in range(len(dataset_to_use)):
        dn = dataset_to_use[i]
        l = label_per_data[datasets_pools.index(dn)]

        for _ in tqdm(range(bootstrap_size)):

            al = ActiveLearning(data_path=f"data/{dn}_data.csv", feature_of_interest=l, size_limit=4000, iterations=10,
                                budget_per_iter=-1, train_label_test_split=(0.45, 0.45, 0.1))

            cur_res = al.run_experiments(methods2use=methods_to_use)
            bootstrap_results[dn].append(cur_res)


    with open(fr"results/results_partial_labeling_4k_s={bootstrap_size}.pkl", 'wb') as file:
        pickle.dump(bootstrap_results, file)

