import copy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random

plt.switch_backend('TkAgg')


def generate_plot(accuracy_scores_dict):
    """
    Generate a plot for the accuracy scores
    """
    for criterion, accuracy_scores in accuracy_scores_dict.items():
        plt.plot(range(1, len(accuracy_scores) + 1), accuracy_scores, label=criterion)

    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Scores for Different Criteria')
    plt.legend()
    plt.grid(True)  # Optional: Adds gridlines to the plot
    plt.show()


class ActiveLearningPipeline:

    def __init__(self, iterations, budget_per_iter, data_path, train_label_test_split: tuple):

        self.model = None  # model

        self.iterations = iterations
        self.budget_per_iter = budget_per_iter

        # read and prepare train data, data to label and test data
        data_df = pd.read_csv(data_path)
        data_df = shuffle(data_df, random_state=42)

        train_df = data_df.iloc[:train_label_test_split[0]]
        label_df = data_df.iloc[train_label_test_split[0]:train_label_test_split[1]]
        test_df = data_df.iloc[train_label_test_split[1]:train_label_test_split[2]]

        train_data = {'x': [np.array(row) for row in train_df.drop('Diabetes', axis=1).to_numpy()],
                      'y': np.array([l for l in train_df['Diabetes']])}

        label_data = {'x': [np.array(row) for row in train_df.drop('Diabetes', axis=1).to_numpy()],
                      'y': np.array([l for l in label_df['Diabetes']])}

        test_data = {'x': [np.array(row) for row in train_df.drop('Diabetes', axis=1).to_numpy()],
                     'y': np.array([l for l in test_df['Diabetes']])}

        self.data = {'train_data': train_data, 'label_data': label_data, 'test_data': test_data}
        self.features = np.array(data_df.drop('Diabetes', axis=1).columns)

    " Sampling Methods "

    def _random_sampling(self, pool_size):
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
        selected_indices = [i for i in selected_indices]
        return selected_indices

    def risk_based_sampling(self, unlabeled_x, y_pred, n_select):
        # Calculate uncertainty (entropy)
        uncertainties = entropy(y_pred.T)

        # Extract important feature columns (BMI, Smoking, Heart Disease...)
        features_list = list(self.features)  # In case self.features is not a list
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
        unlabeled_x = np.array(unlabeled_x)
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
        selected_indices = np.argsort(combined_scores)[-n_select:]
        return selected_indices

    " Main Method "
    def run_pipeline(self, selection_criterion):
        """ Run the active learning pipeline """

        train_data_x = copy.deepcopy(self.data['train_data']['x'])
        train_data_y = copy.deepcopy(self.data['train_data']['y'])

        label_data_x = copy.deepcopy(self.data['label_data']['x'])
        label_data_y = copy.deepcopy(self.data['label_data']['y'])

        test_data_x = copy.deepcopy(self.data['test_data']['x'])
        test_data_y = copy.deepcopy(self.data['test_data']['y'])

        accuracy_scores = []
        for iteration in range(self.iterations):

            if len(label_data_y) < self.budget_per_iter:
                break

            # 1. create and fit model on available train data
            train_x, train_y = train_data_x, train_data_y
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(train_x, train_y)

            # 2. Compute accuracy
            test_x, test_y = test_data_x, test_data_y
            y_pred = self.model.predict(test_x)
            accuracy = np.mean(y_pred == test_y)
            accuracy = round(float(accuracy), 3)
            accuracy_scores.append(accuracy)

            # 3. predict unlabeled data and choose samples to label (get probabilities, not predicted labels)
            unlabeled_x = label_data_x
            y_pred = self.model.predict_proba(unlabeled_x)

            # 3. choose data to be labeled
            if selection_criterion == 'random':
                add_to_train_indices = self._random_sampling(len(y_pred))
            elif selection_criterion == 'uncertainty':
                add_to_train_indices = self._uncertainty_sampling(y_pred)
            elif selection_criterion == 'risk_based':
                add_to_train_indices = self.risk_based_sampling(unlabeled_x, y_pred, n_select=5)
            else:
                raise RuntimeError("unknown method")

            for idx in sorted(add_to_train_indices, reverse=True):
                train_data_x.append(label_data_x.pop(idx))
                train_data_y = np.append(train_data_y, label_data_y[idx])
                label_data_y = np.delete(label_data_y, idx)

        return accuracy_scores


if __name__ == '__main__':

    al = ActiveLearningPipeline(iterations=10, budget_per_iter=10,
                                data_path=r"C:\Users\user\Desktop\converted_data.csv",
                                train_label_test_split=(100, 200, 300))

    sampling_methods_to_try = ['random', 'uncertainty', 'risk_based']

    methods_performance = {}
    for sm in sampling_methods_to_try:
        sm_result = al.run_pipeline(selection_criterion=sm)
        methods_performance[sm] = sm_result

    print(methods_performance)
    generate_plot(methods_performance)
