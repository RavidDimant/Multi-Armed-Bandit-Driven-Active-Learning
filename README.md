# 🌟 Active Learning Project




## 🚀 Description
The **Active Learning Project** aims to explore and evaluate multiple active learning strategies, with a unique emphasis on the **Multi-Armed Bandit (MAB) method**, to enhance the training process of machine learning models. By selecting the most informative data points to be labeled, the project seeks to improve model performance while minimizing the labeling budget.

Unlike traditional approaches, our **MAB-based pipeline** adapts dynamically, choosing the best sampling strategy at each iteration based on previous successes. This adaptive selection not only optimizes labeling efficiency but accelerates model learning by maximizing data impact per label. The repository provides an implementation of various active learning techniques, including **Random Sampling**, **Uncertainty Sampling**, **Diversity Sampling**, and other innovative approaches that complement the MAB strategy.

---

## 📖 Table of Contents
- [🚀 Description](#-description)
- [🔧 Installation](#-installation)
- [📝 Usage](#-usage)
- [✨ Features](#-features)
- [📂 Files](#-files)
- [📜 License](#-license)

---

## 🔧 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/username/Active-Learning-Project.git
   ```

2. Change into the project directory:
   ```bash
   cd Active-Learning-Project
   ```

3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📝 Usage

To run the active learning pipeline, make sure you have a dataset in CSV format. The pipeline expects a file path for the dataset and configurations for data splits and iterations.

To execute the main pipeline:

```bash
python active_learning_pipeline.py
```

You can modify the dataset, features, and sampling methods by changing the parameters in the script or passing them as arguments.

---

## ✨ Features

### 1. 🎯 Active Learning Strategies
The **ActiveLearningPipeline** class provides several strategies for data sampling:

- 🔄 **Random Sampling**: Randomly selects samples without any specific heuristic.
- ❓ **Uncertainty Sampling**: Selects samples the model is least certain about to maximize learning efficiency.
- 🌍 **Diversity Sampling**: Selects diverse samples based on pairwise distance to ensure varied data.
- 📈 **Density-Weighted Uncertainty Sampling**: A combination of density-based selection and uncertainty.
- 🤖 **Query by Committee (QBC)**: Uses multiple models to select samples with the highest disagreement.
- ⚠️ **Risk-Based Sampling**: Weights features by correlation to label interest to enhance sampling efficiency.
- 🎰 **Multi-Armed Bandit (MAB) Strategy**: Uses a multi-armed bandit approach to dynamically select the best sampling method.

### 2. 📊 Custom Visualization Tools
The **visualizations.py** script provides functions to visualize the performance of the active learning strategies:

- 📈 **Accuracy Plots**: Track accuracy improvement across iterations.
- 📊 **Comparison Charts**: Compare scores of different sampling methods on various datasets.

You can generate plots using functions like `generate_plot()` and `plot_all_datasets_results()` to visualize and compare different sampling methods.

---

## 📂 Files
- **📜 active_learning_pipeline.py**: Implements the core active learning pipeline, offering multiple sampling methods and training strategies for different datasets.
- **📊 visualizations.py**: Provides utility functions to generate plots comparing the effectiveness of different active learning methods.
- **🗂 process_data.ipynb**: Jupyter notebook for data preprocessing, exploration, and preparation prior to running the main active learning pipeline.
- **📁 Data/**: Folder containing the processed and cleaned datasets used in the project:
  - **converted_car_data.csv**: Processed car evaluation dataset.
  - **converted_diabetes_data.csv**: Cleaned diabetes dataset.
  - **converted_glass_data.csv**: Processed glass dataset for classification.
  - **converted_wine_data.csv**: Cleaned wine quality dataset.
  - **car_evaluation.csv**: Original car evaluation dataset.
  - **CVD_dataset.csv**: Original cardiovascular dataset.
  - **glass.csv**: Original glass dataset.
  - **winequality-red.csv**: Original red wine quality dataset.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
