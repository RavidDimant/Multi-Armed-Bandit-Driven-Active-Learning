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
- [📊 Results](#-results)
- [📋 Project Overview](#-project-overview)

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

## 📊 Results
The results of our experiments demonstrate the effectiveness of our MAB-driven sampling strategies across a variety of datasets. The following metrics summarize the key findings:

- **Apple Dataset**: LST-MAB achieved a mean accuracy of **83.46** with a standard deviation of **1.90**, which was comparable to other methods like Random MAB (**83.46**) and Margin (**83.88**).
- **Loan Dataset**: The Feature-Based method showed the highest mean accuracy (**90.84**) with LST-MAB achieving a comparable performance of **90.74**.
- **MB Dataset**: The Random MAB strategy outperformed other methods with a mean accuracy of **95.73**, while Vanilla MAB and LST-MAB were close with mean scores of **95.48**.
- **Passenger Dataset**: LST-MAB and Vanilla MAB performed almost identically (**92.03**), showing stability across methods.
- **Diabetes Dataset**: All methods, including LST-MAB and Random, yielded similar results (**86.74**).
- **Employee Dataset**: Random MAB and LST-MAB achieved the highest accuracy (**82.38**), outperforming individual static sampling methods.
- **Shipping and Hotel Datasets**: LST-MAB and Vanilla MAB showed consistent performance across the board, with mean scores in the range of **67.69 - 82.77**.

Overall, the results highlight the robustness of MAB-driven strategies, particularly LST-MAB, in selecting effective sampling methods across diverse datasets, often matching or surpassing individual static methods.

---

## 📋 Project Overview
The **Active Learning Project** focuses on enhancing model efficiency through intelligent sampling. By utilizing Multi-Armed Bandit (MAB) algorithms, the project aims to dynamically adapt the sampling process, selecting the most informative data points for labeling. This approach reduces labeling costs while improving model performance across different datasets.

### Setup Instructions
1. Clone the repository and navigate to the project directory.
2. Install the necessary dependencies using the provided `requirements.txt` file.
3. Ensure you have the appropriate datasets in CSV format.

### Running the Code
To run the pipeline, execute the main script as follows:
```bash
python active_learning_pipeline.py
```
You can configure different parameters for datasets and sampling strategies directly within the script or through command line arguments.

The results and visualizations can be generated using the **visualizations.py** script for a comprehensive understanding of the performance of various sampling methods.
