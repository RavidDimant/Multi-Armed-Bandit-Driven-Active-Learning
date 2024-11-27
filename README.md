# 🌟 Active Learning Project

## 🚀 Description

The **Active Learning Project** explores and evaluates a range of active learning strategies, with a primary focus on the **Multi-Armed Bandit (MAB)** method. The goal is to enhance the training process of machine learning models by intelligently selecting the most informative data points for labeling, thereby improving model performance while minimizing the labeling budget.

Unlike traditional approaches, the MAB-based pipeline dynamically adapts at each iteration, selecting the most effective sampling strategy based on previous results. This adaptive methodology optimizes labeling efficiency and accelerates learning by maximizing the impact of each labeled data point. The repository includes implementations of both standard techniques, such as **Random Sampling**, **Uncertainty Sampling**, and **Diversity Sampling**, and advanced strategies like **Vanilla-MAB** and **Long Short-Term MAB (LST-MAB)**, making it a comprehensive toolkit for active learning research and applications.

---

## 📖 Table of Contents
- [🚀 Description](#-description)
- [🔧 Installation](#-installation)
- [📝 Usage](#-usage)
- [✨ Features](#-features)
- [📂 Files](#-files)
- [📊 Results](#-results)

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

---

## 📝 Usage

### Step 1: **Prepare the Dataset**
1. Open the `process_data.ipynb` Jupyter notebook.
2. Load the raw datasets located in the `Data/` folder.
3. Run the notebook cells sequentially to clean and preprocess the datasets:
   - Handle missing values.
   - Normalize or scale numerical features to ensure consistency.
   - Convert categorical features to numeric representations, such as one-hot encoding or label encoding.
   - Save the processed datasets to the `Data/` folder for further use (e.g., `converted_car_data.csv`, `converted_diabetes_data.csv`).

### Step 2: **Train the Model**
1. Open the `active_learning.py` script.
2. Configure the desired parameters for your experiment:
   - Specify the dataset to use (e.g., `converted_diabetes_data.csv`).
   - Set hyperparameters such as the number of iterations, train/test split, and budget per iteration.
   - Choose sampling methods or pipelines to evaluate, such as **uncertainty sampling**, **margin sampling**, or **MAB strategies**.
3. Run the script using the following command in your terminal:
   ```bash
   python active_learning.py
   ```
4. During execution, the script will:
   - Train a machine learning model (e.g., Random Forest or Logistic Regression).
   - Apply the selected active learning strategies.
   - Save the training results and predictions for later analysis.

### Step 3: **Analyze Results**
1. Open the `results_analysis.ipynb` Jupyter notebook.
2. Load the results generated during the training phase.
3. Run the notebook cells to:
   - Compare the performance of different sampling strategies.
   - Generate visualizations such as accuracy plots, bar charts, and comparison tables.
   - Analyze key metrics like final accuracy, mean accuracy across iterations, and accuracy improvements over the baseline.
4. Save or export the generated visualizations and findings as needed for reports or presentations.

By following these steps, you can effectively preprocess data, train models with active learning strategies, and evaluate the results to determine the most effective sampling approach for your dataset.

---

## ✨ Features

### 1. 🎯 Active Learning Strategies

The `ActiveLearningPipeline` class provides several strategies for data sampling:

- 🔄 **Random Sampling**: Randomly selects samples without any specific heuristic.
- ❓ **Uncertainty Sampling**: Selects samples the model is least certain about to maximize learning efficiency.
- 🌍 **Diversity Sampling**: Selects diverse samples based on pairwise distance to ensure varied data.
- 📈 **Density-Weighted Uncertainty Sampling**: Combines density-based selection and uncertainty for improved sampling efficiency.
- 🤖 **Query by Committee (QBC)**: Utilizes multiple models to select samples with the highest disagreement, promoting diverse learning.
- ⚠️ **Risk-Based Sampling**: Weights features by correlation to label interest to enhance sampling efficiency in specific tasks.
- 🎰 **Multi-Armed Bandit (MAB) Strategy**: Dynamically selects the best sampling method using bandit algorithms, optimizing for robust performance.

### 2. 🌟 Adaptive Sampling
- Dynamically adapts the sampling process using **MAB-based strategies**, ensuring consistent and effective performance across a variety of datasets. Balances exploration and exploitation to maximize learning impact per label.

### 3. 💡 Diverse Strategies
- Offers a range of standard methods like **uncertainty sampling**, **margin sampling**, and **diversity sampling**, alongside advanced pipelines such as **Vanilla-MAB** and **Long Short-Term MAB (LST-MAB)** for sophisticated active learning needs.

### 4. 📊 Comprehensive Visualization
- Provides detailed analysis tools for visualizing and comparing sampling strategies, including:
  - 📈 **Accuracy Plots**: Track accuracy trends across iterations.
  - 📊 **Comparison Charts**: Highlight the performance of different sampling strategies.
  - 🗂 **Bar and Box Plots**: Display accuracy improvements and variability for in-depth analysis. 

These features make the project a versatile and powerful framework for evaluating and enhancing active learning processes.
---

## 📂 Files
- **📜 active_learning.py**: Implements the core active learning pipeline, offering multiple sampling methods and training strategies for different datasets.
- **📊 visualizations.py**: Provides utility functions to generate plots comparing the effectiveness of different active learning methods.
- **📜 multi_arm_bandit.py**: Contains the implementation of the Multi-Armed Bandit (MAB) framework, which dynamically selects the most effective sampling strategy during active learning.
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


 ![Image Alt](https://github.com/RavidDimant/Multi-Armed-Bandit-Driven-Active-Learning/blob/main/results/Last%20Iterations%20Result.png?raw=true)

**Last Iterations Result** showcase the performance improvements of various sampling methods across multiple datasets in the final iteration of the active learning process. Adaptive methods like **LST-MAB** and **Vanilla MAB** consistently achieve top performance, particularly in datasets like **Apple**, **Hotel**, and **Shipping**, where their dynamic selection of strategies maximizes learning efficiency. Static methods, such as **Margin Sampling** and **Feature-Based Sampling**, demonstrate competitive results in specific datasets like **Loan** and **MB**, but lack the versatility of the MAB approaches. The chart also highlights that simpler methods, like **Random Sampling**, offer modest improvements but often fall behind more targeted strategies. This emphasizes the strength of adaptive frameworks like **MAB** in handling diverse data characteristics and achieving robust results across various datasets.

---
