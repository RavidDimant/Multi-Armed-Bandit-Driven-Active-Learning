# 🌟 Multi Armed Bandit Driven Active Learning Project

## 🚀 Description

The **Multi Armed Bandit Driven Active Learning Project** explores and evaluates a range of active learning strategies, with a primary focus on the **Multi-Armed Bandit (MAB)** method. The goal is to enhance the training process of machine learning models by intelligently selecting the most informative data points for labeling, thereby improving model performance while minimizing the labeling budget.

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

### Disclaimer:
If you already have the results file (results_partial_labeling_4k_s=100.pkl) available, you can directly run results_analysis.ipynb without completing Steps 1 and 2. This file contains the pre-computed results, allowing you to skip data preprocessing and model training. If the file is unavailable, proceed with the conventional method outlined below.


### Step 1: **Prepare the Dataset**
1. Open the `process_data.ipynb` Jupyter notebook.
2. Load the raw datasets located in the `Data/` folder. If you have another dataset you want to explore, upload it to also to the notebook.
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

Note that the execution will take a lot time because we iterate over all datasets.

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

- 🎲 **Random Sampling**: Randomly selects samples without any specific heuristic.
- ❓ **Uncertainty Sampling**: Selects samples the model is least certain about to maximize learning efficiency.
- 🌍 **Diversity Sampling**: Selects diverse samples based on pairwise distance to ensure varied data.
- 📈 **Density-Weighted Uncertainty Sampling**: Combines density-based selection and uncertainty for improved sampling efficiency.
- ⚖️ **Margin Sampling**: Selects samples closest to the decision boundary, maximizing the model's learning potential.
- 🤖 **Query by Committee (QBC)**: Utilizes multiple models to select samples with the highest disagreement, promoting diverse learning.
- 🔄 **Metropolis Hastings Sampling**: Applies a probabilistic model to iteratively select samples based on acceptance criteria, balancing exploration and exploitation.
- ⚠️ **Feature-Based Sampling**: Weights features by correlation to label interest to enhance sampling efficiency in specific tasks.
- 🎰 **Multi-Armed Bandit (MAB) Strategy**: Dynamically selects the best sampling method using bandit algorithms, optimizing for robust performance.

### 2. 🏹 Vanilla Multi-Armed Bandit (MAB)
- Employs a **single bandit** to manage the sampling process across all iterations.
- Uses Upper Confidence Bound (UCB) algorithms to balance exploration (trying lesser-used sampling methods) and exploitation (selecting the best-performing methods).
- Demonstrates robust performance across datasets by dynamically adjusting to data characteristics and improving sampling efficiency over time.

### 3. ⚡ Long Short-Term MAB (LST-MAB)
- Extends the vanilla MAB with **short-term and long-term memory mechanisms**, allowing for adaptive switching between strategies based on recent and historical performance.
- Leverages **KL divergence** to compare short-term and long-term behaviors, ensuring a dynamic and context-aware sampling approach.
- Excels in datasets with changing distributions or where the best strategy varies significantly over time, making it highly effective for diverse applications.

### 4. 💡 Diverse Strategies
- Offers a range of standard methods such as **Random Sampling**, **Uncertainty Sampling**, **Diversity Sampling**, **Density-Weighted Uncertainty Sampling**, **Margin Sampling**, **Query by Committee (QBC)**, and **Metropolis Hastings Sampling**.
- Incorporates advanced adaptive pipelines like **Vanilla-MAB** and **Long Short-Term MAB (LST-MAB)** for sophisticated active learning tasks.

### 5. 📊 Comprehensive Visualization
- Provides detailed analysis tools for visualizing and comparing sampling strategies, including:
  - 📈 **Accuracy Plots**: Track accuracy trends across iterations.
  - 📊 **Comparison Charts**: Highlight the performance of different sampling strategies.
  - 🗂 **Bar and Box Plots**: Display accuracy improvements and variability for in-depth analysis.

These features make the project a versatile and powerful framework for evaluating and enhancing active learning processes across diverse datasets.

---

## 📂 Files

- **📜 active_learning.py**: Implements the core active learning pipeline, offering multiple sampling methods and training strategies for different datasets.
- **📊 visualizations.py**: Provides utility functions to generate plots comparing the effectiveness of different active learning methods.
- **📜 multi_arm_bandit.py**: Contains the implementation of the Multi-Armed Bandit (MAB) framework, which dynamically selects the most effective sampling strategy during active learning.
- **🗂 process_data.ipynb**: Jupyter notebook for data preprocessing, exploration, and preparation prior to running the main active learning pipeline.
- **📁 Results/**: Folder containing datasets used in the project:
  - **results_partial_labeling_4k_s=100.pkl**: Pre-computed results file for direct use in analysis, bypassing data preparation and training.
- **📁 Data/**: Folder containing datasets used in the project:
  - **apple_data.csv**: Dataset for analyzing quality-related tasks for Apple products.
  - **diabetes_data.csv**: Dataset for diabetes classification tasks.
  - **employee_data.csv**: Dataset focusing on employee retention analysis.
  - **hotel_data.csv**: Dataset analyzing hotel reservation behavior.
  - **loan_data.csv**: Dataset for loan status prediction.
  - **mb_data.csv**: Dataset for preferences between mountains and beaches.
  - **passenger_data.csv**: Dataset evaluating passenger satisfaction levels.
  - **shipping_data.csv**: Dataset analyzing shipping performance outcomes.
  - **wine_data.csv**: Dataset for wine quality classification.
  - **classification/**: Subdirectory containing original or intermediate processed datasets:
    - **Employee.csv**: Original employee dataset.
    - **apple_quality.csv**: Dataset for Apple quality analysis.
    - **diabetes.csv**: Original diabetes dataset.
    - **hotel_reservations.csv**: Dataset for hotel reservation analysis.
    - **loan_data.csv**: Original loan dataset.
    - **mountains_vs_beaches_preferences.csv**: Dataset for analyzing preferences between mountains and beaches.
    - **passenger_satisfaction.csv**: Original dataset for passenger satisfaction analysis.
    - **shipping.csv**: Dataset for analyzing shipping data.
    - **wine_quality.csv**: Original wine quality dataset.

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

 ![Image Alt](https://github.com/RavidDimant/Multi-Armed-Bandit-Driven-Active-Learning/blob/main/results/Last%20Iterations%20Result.png?raw=true)

**Last Iterations Result** showcase the performance improvements of various sampling methods across multiple datasets in the final iteration of the active learning process. Adaptive methods like **LST-MAB** and **Vanilla MAB** consistently achieve top performance, particularly in datasets like **Apple**, **Hotel**, and **Shipping**, where their dynamic selection of strategies maximizes learning efficiency. Static methods, such as **Margin Sampling** and **Feature-Based Sampling**, demonstrate competitive results in specific datasets like **Loan** and **MB**, but lack the versatility of the MAB approaches. The chart also highlights that simpler methods, like **Random Sampling**, offer modest improvements but often fall behind more targeted strategies. This emphasizes the strength of adaptive frameworks like **MAB** in handling diverse data characteristics and achieving robust results across various datasets.

---

 ![Image Alt](https://github.com/RavidDimant/Multi-Armed-Bandit-Driven-Active-Learning/blob/main/results/Mean%20Iteration%20Results.png?raw=true)

The **Mean Iteration Results** graph highlights the average performance of sampling methods across multiple iterations for different datasets. Adaptive approaches like **LST-MAB** and **Vanilla MAB** continue to dominate, showcasing their ability to maintain consistently high performance over time, particularly in datasets like **Hotel**, **Apple**, and **MB**. Static methods, such as **Margin Sampling** and **Feature-Based Sampling**, also perform well in specific datasets, such as **Loan** and **Shipping**, but show a drop in adaptability compared to MAB-based methods. Interestingly, **Uncertainty Sampling** and **Density-Weighted Uncertainty Sampling** provide competitive performance in moderately complex datasets, while simpler methods like **Random Sampling** achieve only marginal gains across most datasets. These findings reaffirm the importance of adaptive sampling strategies, especially in scenarios where consistent performance across iterations is critical.

---

 ![Image Alt](https://github.com/RavidDimant/Multi-Armed-Bandit-Driven-Active-Learning/blob/main/results/Overall%20Performance%20-%20Mean%20of%20Iterations.png?raw=true)

The **Overall Performance - Mean of Iterations** chart summarizes the aggregated accuracy improvements across all datasets for each sampling method. Adaptive methods, such as **Vanilla MAB** and **LST-MAB**, stand out with high performance, achieving improvements of 5.14% and 4.94%, respectively, highlighting their effectiveness in dynamic environments. **Margin Sampling** slightly outperforms other static methods with an aggregated improvement of 6.33%, demonstrating its utility in specific scenarios. **Uncertainty Sampling** and **QBC Sampling** also perform well, with improvements of 6.31% and 3.13%, showcasing their ability to handle uncertainty-driven selections. Conversely, simpler methods like **Random Sampling** (1.04%) and **Diversity Sampling** (0.92%) show limited improvements, reflecting their lack of precision. The results underscore the value of adaptive strategies, particularly MAB-based approaches, in maintaining robust and consistent performance across diverse datasets.

---

 ![Image Alt](https://github.com/RavidDimant/Multi-Armed-Bandit-Driven-Active-Learning/blob/main/results/Overall%20Performance%20-%20Last%20Iteration.png?raw=true)


The **Overall Performance - Last Iteration** chart illustrates the aggregated accuracy improvements across all datasets for each sampling method in the final iteration. **LST-MAB** and **Vanilla MAB** demonstrate the highest overall performance, with improvements of 5.44% and 5.43%, respectively, showcasing their consistency and adaptability even in the final stages of the learning process. **Margin Sampling** and **Uncertainty Sampling** also perform strongly, achieving aggregated improvements of 5.32% and 5.02%, indicating their effectiveness in scenarios requiring precise boundary decision-making. On the other hand, simpler methods like **Random Sampling** and **Diversity Sampling** exhibit relatively modest improvements of 1.51% and 0.92%, reflecting their limited ability to prioritize impactful data points. The results highlight the superiority of adaptive sampling strategies, particularly MAB-based methods, in maintaining robust and reliable performance throughout the active learning process.

